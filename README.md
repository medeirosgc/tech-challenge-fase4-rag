# Voice of Customer Intelligence com RAG — Olist

**Tech Challenge — Fase 4 | Pós-Tech FIAP IA**

Solução baseada em **Retrieval-Augmented Generation (RAG)** que permite a gestores e analistas fazer perguntas em linguagem natural sobre as avaliações de clientes da base *Brazilian E-Commerce Public Dataset by Olist* e receber respostas **fundamentadas nas avaliações reais**, com as evidências exibidas e recusa explícita quando não há base suficiente para responder.

---

## 1. Problema de negócio

Empresas de e-commerce recebem milhares de avaliações. A nota (1 a 5) mostra *se* o cliente está satisfeito, mas o *porquê* está nos comentários em texto — e ler milhares deles manualmente não escala. Esta solução transforma os comentários em uma base de conhecimento consultável: o gestor pergunta "quais são os principais problemas relatados?" e recebe uma síntese com as avaliações que sustentam cada ponto.

## 2. Arquitetura

```
Pergunta do usuário
   ↓
Processamento da pergunta  → limpeza + detecção de intenção (filtro por nota)
   ↓
Embedding da pergunta      → paraphrase-multilingual-MiniLM-L12-v2 (mesmo modelo dos documentos)
   ↓
Recuperação                → ChromaDB (índice HNSW, similaridade de cosseno, filtros por metadados)
   ↓
Seleção                    → descarta avaliações abaixo do limiar de similaridade
   ↓                          (< 3 evidências → recusa sem chamar o LLM)
Construção do contexto     → avaliações numeradas [n] + nota, entrega e categoria, dentro de <avaliacoes>
   ↓
Geração                    → Gemini (temperature 0.2), prompt com regras de fundamentação
   ↓
Resposta + evidências      → texto com citações [n] + tabela das avaliações citadas (review_id, nota, similaridade)
```

## 3. Tecnologias e bibliotecas

| Componente | Tecnologia | Por que |
|---|---|---|
| Ambiente | Google Colab (GPU T4) | Gratuito, sem instalação local, GPU para os embeddings |
| Preparação dos dados | pandas | Padrão de mercado para manipulação tabular |
| Embeddings | `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2` | Embeddings de frase multilíngues (bom em PT-BR), gratuitos, rodam localmente, 384 dimensões |
| Banco vetorial | ChromaDB | Persistência em disco, índice HNSW, filtros por metadados, sem servidor |
| LLM | Google Gemini Flash (`google-genai`, modelo `gemini-flash-latest`, reserva `gemini-flash-lite-latest`) | Plano gratuito suficiente para o volume do projeto; boa qualidade em português |
| Download da base | `kagglehub` | Baixa o dataset direto do Kaggle dentro do Colab |

## 4. Preparação da base de conhecimento

| Etapa | Decisão | Justificativa |
|---|---|---|
| **Seleção dos dados** | `order_reviews` (texto e nota) + `orders` (datas de entrega) + `order_items`/`products` (categoria) | Só o que ajuda a responder perguntas sobre a voz do cliente; pagamentos, vendedores e geolocalização foram descartados |
| **Valores ausentes** | Avaliações sem nenhum texto são descartadas; título ausente vira string vazia; metadados ausentes viram `sem_info` / `sem_categoria` | O desafio é sobre o conteúdo textual — avaliação sem texto não tem o que recuperar |
| **Limpeza** | Mínima: apenas quebras de linha e espaços repetidos | Embeddings e LLMs entendem linguagem natural. Stemming ou remoção de stopwords apagariam palavras como "não", invertendo o sentido ("não gostei" → "gostei") |
| **Título + comentário** | Concatenados quando o título traz informação nova | O título frequentemente resume a queixa ("Produto com defeito") |
| **Textos curtos** | Descartados comentários com menos de 3 palavras | "ok", "top" carregam pouca informação e poluem a recuperação |
| **Duplicatas** | Uma linha por `review_id` | Algumas avaliações aparecem ligadas a mais de um pedido |
| **Organização dos documentos** | 1 avaliação = 1 documento, **sem chunking** | Os comentários são curtos; dividi-los quebraria o sentido. Cada avaliação já é uma unidade completa de evidência |
| **Metadados** | `nota`, `entrega` (no_prazo/atrasada/sem_info), `mes_compra`, `categoria` | Permitem filtrar a busca antes da similaridade (ver seção 7) |

**Números da base (saída real do notebook):**

| Etapa | Quantidade |
|---|---|
| Linhas em `order_reviews` | 99.224 |
| Avaliações únicas (`review_id`) | 98.410 |
| Com algum texto (título ou comentário) | 42.370 |
| **Base final indexada (≥ 3 palavras)** | **36.040** |

- 58,7% das avaliações não têm comentário e 88,3% não têm título.
- Embeddings: matriz 36.040 × 384 gerada em 22 s (GPU T4).
- Metadado de entrega: 30.693 no prazo, 3.574 atrasadas, 1.773 sem informação. **Nota média: 1,85 nas atrasadas vs 3,89 no prazo** — o que confirma o valor do metadado para filtrar a busca.

## 5. Estratégia do pipeline RAG

1. **Transformação da pergunta:** a pergunta é limpa e convertida em embedding com o **mesmo modelo** dos documentos (condição para os vetores serem comparáveis). Regras simples detectam a intenção e aplicam filtro de metadado: perguntas sobre reclamações/insatisfação buscam nas notas 1–2; sobre elogios, nas notas 4–5.
2. **Recuperação semântica:** busca dos `K = 40` vizinhos mais próximos no ChromaDB (índice HNSW). A busca é por significado: "chegou atrasado" recupera "demorou muito para entregar".
3. **Seleção:** cálculo da similaridade de cosseno e descarte das avaliações abaixo do limiar (`LIMIAR_SIM = 0,45`, calibrado comparando perguntas do escopo com perguntas fora do escopo — ver tabela abaixo).
4. **Construção do contexto:** as avaliações selecionadas são numeradas `[1]…[n]` com nota, status da entrega e categoria, e colocadas dentro de tags `<avaliacoes>`, separadas das instruções.
5. **Geração:** o Gemini recebe instruções para usar **somente** o contexto, citar `[n]` em cada afirmação, informar quantas avaliações mencionam cada tema e responder `SEM_EVIDENCIA` quando o contexto não sustenta a resposta. `temperature = 0.2` para respostas consistentes.

**Calibração do limiar** (média da similaridade das 10 avaliações mais próximas):

| Escopo | Pergunta | Média top-10 |
|---|---|---|
| dentro | O produto chegou quebrado | 0,973 |
| dentro | A entrega atrasou muito | 0,898 |
| dentro | Gostei muito do atendimento | 0,954 |
| fora | Qual a capital da França? | 0,370 |
| fora | Como calcular a derivada de x ao quadrado? | 0,362 |
| fora | Receita de bolo de cenoura | 0,509 |

O limiar 0,45 separa os grupos. Casos limítrofes como "receita de bolo" (0,51) passam pela 1ª barreira e são tratados pela 2ª (instrução `SEM_EVIDENCIA` no prompt).

**Por que K = 40?** Perguntas como "quais problemas aparecem com maior frequência" exigem uma amostra maior que os 3–5 trechos de um RAG típico, para que o modelo identifique temas recorrentes e conte quantas avaliações mencionam cada um.

## 6. Fundamentação e tratamento de falta de evidência

Três barreiras contra respostas sem suporte:

| Barreira | Onde | Situação coberta |
|---|---|---|
| Menos de 3 avaliações acima do limiar → recusa **sem chamar o LLM** | Recuperação | Pergunta fora do escopo; nenhuma avaliação relevante |
| Prompt manda responder `SEM_EVIDENCIA` | Geração | Evidências recuperadas insuficientes para a pergunta |
| Citações `[n]` + tabela com `review_id`, nota e similaridade das avaliações citadas | Saída | Rastreabilidade: o usuário confere cada insight |

**Segurança:** as avaliações são texto escrito por terceiros. O prompt as trata explicitamente como *dados*, não instruções, e elas ficam isoladas em tags — mitigação de *prompt injection* indireta.

## 7. Desafio adicional — metadados

Os metadados restringem a busca **antes** da similaridade. Isso melhora a recuperação porque o contexto enviado ao LLM contém só avaliações do recorte desejado (menos ruído), e permite perguntas que a semântica sozinha não resolve, como "pedidos que de fato chegaram atrasados" (calculado comparando a data de entrega com a data estimada).

Exemplos implementados:
- Problemas em avaliações com nota baixa → `nota ≤ 2` (automático pela intenção da pergunta)
- Temas em entregas insatisfatórias → `entrega = atrasada` **e** `nota ≤ 2`
- Elogios em uma categoria → `categoria = cama_mesa_banho` **e** `nota ≥ 4`

## 8. Exemplos de perguntas e respostas (resultados reais)

| Pergunta | Filtro aplicado | Evidências acima do limiar | Avaliações citadas | Status |
|---|---|---|---|---|
| Quais são os principais problemas relatados pelos clientes? | nota ≤ 2 (automático) | 40 | 36 | respondida |
| O que os clientes relatam sobre problemas de entrega? | nota ≤ 2 (automático) | 40 | 39 | respondida |
| Quais aspectos da experiência de compra são mais elogiados? | nota ≥ 4 (automático) | 40 | 24 | respondida |
| Quais padrões nas avaliações de clientes insatisfeitos? | nota ≤ 2 (automático) | 40 | 39 | respondida |
| Existem padrões recorrentes sobre a qualidade dos produtos? | — | 40 | 40 | respondida |
| Quais temas aparecem nas avaliações de entregas insatisfatórias? | entrega = atrasada **e** nota ≤ 2 | 40 | 40 | respondida |
| Qual a capital da França? *(fora do escopo)* | — | 1 | 0 | **recusada antes do LLM** |

**Exemplo de resposta completa** — *"Quais são os principais problemas relatados pelos clientes em suas avaliações?"*

> **Resposta:** Os principais problemas relatados concentram-se em atrasos e não recebimento dos pedidos, além da insatisfação com itens entregues com defeito, divergentes ou de baixa qualidade. Os consumidores também apontam falhas críticas de comunicação e desinformação sobre o andamento das compras, somadas a dificuldades de acesso aos canais de atendimento. Diante desses transtornos, emerge nas mensagens uma queixa recorrente de descaso e falta de respeito com o consumidor.
>
> **Principais pontos:**
> - **Sentimento de desrespeito e falta de consideração:** 11 de 40 avaliações ([2][4][6][10][19][21][22][23][26][33][34]).
> - **Atraso na entrega e produtos não recebidos:** 10 de 40 avaliações ([10][16][17][20][21][23][25][30][33][36]).
> - **Baixa qualidade ou defeito no produto:** 9 de 40 avaliações ([7][12][17][28][31][35][37][38][40]).
> - **Recebimento de produto incorreto ou divergente:** 7 de 40 avaliações ([1][5][13][24][29][32][40]).
> - **Falta de informação e comunicação sobre os pedidos:** 6 de 40 avaliações ([2][20][25][26][27][30]).
> - **Atendimento ineficiente e canais de contato falhos:** 6 de 40 avaliações ([8][9][21][25][29][39]).
> - **Venda sem estoque e cancelamento unilateral:** 2 de 40 avaliações ([2][15]).
>
> **Limitações:** Esta análise baseia-se estritamente na amostra de 40 avaliações fornecidas, não cobrindo a base integral de clientes.

Evidências exibidas pelo sistema para essa resposta (amostra):

| ref | review_id | nota | entrega | texto |
|---|---|---|---|---|
| [1] | db2ad68a772d476d92bae6d8c9a8eb18 | 2 | no_prazo | Produto recebido divergente ao comprado, gerando frustração e perda de oferta de outras opções analisadas. |
| [2] | 0d592da0b11f266efd97a16956ff6417 | 1 | sem_info | compra efetuada, débito autorizado, pedido cancelado sem explicações. Extrema falta de respeito e consideração… |
| [5] | c073a54bbfbbd0282b00f3d50ccdd00f | 1 | no_prazo | reclamação. produto incorreto |
| [7] | a6ae1375e10c5bd2f0984336dbfaf998 | 2 | no_prazo | O produto está com problemas (defeito). |
| [15] | 39e3220236b9d66fb03ab618ed8e3477 | 2 | sem_info | Ponto negativo foi a venda de produto que não possui em estoque. O que frustra o consumidor. |

**Resposta à pergunta fora do escopo** — *"Qual a capital da França?"*

> Não encontrei avaliações suficientes na base para responder com segurança. A pergunta pode estar fora do escopo (avaliações de clientes do Olist) ou não há relatos relevantes sobre esse tema.

As respostas completas de todas as perguntas estão nas saídas do notebook.

## 9. Como reproduzir

1. Abra o notebook no Google Colab.
2. Em *Ambiente de execução → Alterar o tipo de ambiente de execução*, escolha **GPU T4**.
3. Crie uma chave gratuita da Gemini API em <https://aistudio.google.com> (*Get API key → Criar chave*; não é necessário cadastrar faturamento).
4. No ícone de chave 🔑 (*Secrets*) do Colab, adicione um secret chamado `FASE4` com a chave e ative o **acesso ao notebook**.
5. Execute as células em ordem (*Ambiente de execução → Executar tudo*). A base é baixada automaticamente do Kaggle via `kagglehub` (<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>).
6. Se o modelo estiver indisponível, ajuste `MODELO_LLM` para outro modelo "flash" listado pela célula de verificação de modelos.

Dependências: ver `requirements.txt`. Tempo total de execução: cerca de 10 minutos (a maior parte são chamadas ao LLM).

## 10. Principais decisões técnicas e limitações

**Decisões**
- **Embeddings locais gratuitos** em vez de API paga: sem custo, sem limite de uso e qualidade adequada para textos curtos em português.
- **Sem chunking:** avaliações são curtas e autocontidas.
- **Pré-processamento mínimo:** preserva negações e nuances que o modelo de embeddings e o LLM sabem interpretar.
- **Recusa antes do LLM:** perguntas sem evidência nem chegam ao modelo — evita alucinação e economiza chamadas.
- **SDK direto do Gemini + ChromaDB** em vez de um framework de orquestração: o pipeline é linear e curto, então chamar as bibliotecas diretamente deixa cada etapa explícita e fácil de explicar.

**Limitações**
- As contagens ("8 de 40 avaliações") descrevem a **amostra recuperada**, não a base inteira — são indicativas, não estatísticas.
- O limiar de similaridade foi calibrado manualmente com poucos exemplos.
- Em pedidos com vários itens, considera-se a categoria do primeiro item.
- O plano gratuito do Gemini teve instabilidade (erros 503 de alta demanda) durante os testes. O código faz retentativas com espera e pode alternar para um modelo reserva (Flash-Lite).

**Próximos passos possíveis:** busca híbrida (BM25 + vetorial), re-ranking com cross-encoder, avaliação automática (RAGAS) e interface web (Streamlit).

---

**Autor:** Fernando Medeiros — Pós-Tech FIAP IA
