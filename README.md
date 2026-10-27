# AI Image Generator API

[![Verificação estática Python](https://github.com/Diogocvlh/ai-image-generator-api/actions/workflows/python-checks.yml/badge.svg?branch=main)](https://github.com/Diogocvlh/ai-image-generator-api/actions/workflows/python-checks.yml)

API experimental em Python para gerar imagens a partir de descrições de texto. O projeto integra FastAPI, PyTorch e Diffusers em uma execução local com GPU NVIDIA.

**Estado:** protótipo de portfólio. A documentação e a sintaxe foram revisadas; a CI verifica apenas a sintaxe, sem carregar o modelo; a inferência em GPU não foi validada nesta revisão. O projeto não é um serviço de produção.

## O que está implementado

- Endpoint `POST /gerar` com validação do corpo da requisição.
- Autenticação por chave no cabeçalho `x-api-key`, configurada fora do código.
- Parâmetros de resolução, passos, guidance e seed opcional.
- Pipeline Stable Diffusion com scheduler DPM-Solver.
- Salvamento dos resultados em `outputs/`, com nome UUID.
- Tratamento de falta de memória da GPU e de falhas de geração.
- Documentação interativa do FastAPI em `/docs`.

O servidor retorna o caminho do arquivo salvo. Não existe endpoint para baixar a imagem, interface web, fila de tarefas ou registro automático das gerações em banco.

## Tecnologias

| Camada | Ferramentas |
| --- | --- |
| API e validação | FastAPI, Pydantic, Uvicorn |
| Inferência | PyTorch, Diffusers, Transformers, Accelerate |
| Imagens e pesos | Pillow, Safetensors |
| Configuração | Variáveis de ambiente, python-dotenv |

O modelo configurado é [Realistic Vision V6.0 B1](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE). Os pesos são de terceiros e não estão incluídos neste repositório. Consulte a licença e as recomendações do autor; a versão é experimental e o autor recomenda um VAE adicional para melhorar a qualidade. O código atual não configura um VAE externo.

## Pré-requisitos

- Python 3.10 ou superior, em versão compatível com os pacotes escolhidos.
- GPU NVIDIA com driver e instalação do PyTorch compatíveis com CUDA.
- Memória de GPU suficiente para o modelo e a resolução escolhida. Não há medição de VRAM mínima validada neste projeto.
- Acesso à internet no primeiro carregamento e espaço em disco para o cache dos pesos.

Esta implementação usa `float16` e `.to("cuda")`; não possui alternativa em CPU ou Apple Silicon. A instalação do PyTorch depende do seu sistema: use o [seletor oficial](https://pytorch.org/get-started/locally/), em vez de copiar uma versão de CUDA de outro computador.

## Instalação no Windows / PowerShell

Na pasta deste repositório:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instale o PyTorch com CUDA usando o comando fornecido pelo seletor oficial. Depois:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
```

O resultado precisa indicar `True`. O manifesto lista as dependências diretas e faixas de versão; não é um lock de ambiente certificado em GPU. Confira também as [instruções oficiais do Diffusers](https://huggingface.co/docs/diffusers/installation).

### Chave de acesso

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copie a chave gerada para `API_KEY` no arquivo `.env`, mantendo esse arquivo somente no seu computador. O servidor recusa iniciar se a chave tiver menos de 32 caracteres. Variáveis de ambiente já definidas no terminal têm prioridade sobre `.env`.

### Iniciar o servidor

```powershell
python main.py
```

O modelo é carregado antes de o servidor ficar disponível. O primeiro início pode demorar para baixar os pesos; não use `--reload` ou vários workers nesta versão, pois cada processo carrega sua própria cópia do modelo.

Após a inicialização, abra [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). O comando padrão escuta apenas no computador local. Para encerrar, pressione `Ctrl+C`.

Otimização opcional: `ENABLE_XFORMERS=true` habilita xFormers se uma versão compatível estiver instalada. Essa opção fica desabilitada por padrão. Consulte a [documentação do Diffusers sobre xFormers](https://github.com/huggingface/diffusers/blob/main/docs/source/en/optimization/xformers.md) antes de instalar, porque a compatibilidade depende do PyTorch e da plataforma.

## Uso da API

### `POST /gerar`

Cabeçalhos: `Content-Type: application/json` e `x-api-key` com sua chave local.

| Campo | Padrão | Validação |
| --- | --- | --- |
| `prompt` | obrigatório | Entre 3 e 500 caracteres |
| `width` | `512` | Entre 64 e 768; múltiplo de 8 |
| `height` | `512` | Entre 64 e 768; múltiplo de 8 |
| `steps` | `40` | Entre 1 e 60 |
| `guidance` | `7.5` | Número finito entre 1 e 20 |
| `seed` | `null` | Inteiro entre 0 e 9.223.372.036.854.775.807, quando informado |

Exemplo em um segundo terminal PowerShell, na mesma pasta e com o ambiente virtual ativado:

```powershell
$apiKey = python -c "from dotenv import get_key; print(get_key('.env', 'API_KEY'))"
$headers = @{ 'x-api-key' = $apiKey }
$body = @{
    prompt = 'A lighthouse on a rocky coast at sunrise'
    width = 512
    height = 512
    steps = 30
    guidance = 7.5
    seed = 42
} | ConvertTo-Json

Invoke-RestMethod -Uri 'http://127.0.0.1:8000/gerar' `
    -Method Post -Headers $headers -ContentType 'application/json' -Body $body
```

Se você definiu a chave diretamente no ambiente, use `$headers = @{ 'x-api-key' = $env:API_KEY }` no terminal que possuir essa variável.

Formato da resposta de sucesso:

```json
{
  "status": "success",
  "file": "outputs/00000000-0000-4000-8000-000000000000.png",
  "width": 512,
  "height": 512,
  "steps": 30,
  "guidance": 7.5,
  "seed": 42
}
```

O UUID acima é ilustrativo. A seed permite controlar o gerador, mas não garante imagens idênticas entre placas, versões de bibliotecas ou configurações diferentes. O servidor acrescenta instruções de estilo ao prompt e aplica um prompt negativo fixo.

### Respostas de erro

| Código | Situação |
| --- | --- |
| `401` | Chave ausente ou inválida, para uma requisição com corpo válido |
| `422` | Campo ausente, tipo inválido ou parâmetro fora dos limites |
| `500` | Falta de memória da GPU ou falha durante a geração |

Falhas de configuração, ausência de CUDA ou problemas ao carregar o modelo impedem a inicialização. Detalhes de exceções de geração ficam nos logs do servidor; a resposta HTTP usa uma mensagem genérica.

## Organização

```text
.
├── main.py          # Configuração, pipeline e endpoint
├── database.py      # Criação independente de uma tabela SQLite
├── requirements.txt # Dependências diretas
├── .env.example     # Configuração sem credenciais
└── outputs/         # Imagens geradas localmente; ignoradas pelo Git
```

`database.py` é um experimento separado de criação da tabela `geracoes`. Ele não é chamado pela API e não registra requisições. Bancos locais não devem ser publicados como parte do código-fonte.

## Limites e próximos passos

O protótipo usa uma chave compartilhada e processamento síncrono. Ainda não há usuários, permissões, limitação de requisições, controle de concorrência da GPU, limpeza automática dos arquivos ou testes de carga. Execute uma geração por vez em ambiente local. Antes de oferecer acesso público, serão necessários esses controles, HTTPS e testes em hardware real.

A pipeline preserva os mecanismos fornecidos pelo modelo carregado; isso não constitui um sistema completo de moderação. Avaliações de qualidade, desempenho e compatibilidade das dependências ainda precisam ser realizadas com GPU.

Próximas evoluções: separar configuração e inferência, adicionar testes sem GPU com um adaptador de pipeline, implementar fila de tarefas e download autenticado, e decidir se o histórico em banco fará parte do produto.
