# Corretor de Cartões Resposta

Lê cartões do modelo Gabaritech (4 marcadores xadrez nos cantos), corrige pelo gabarito e gera a planilha de notas.
Nenhum cartão é enviado para fora: a leitura acontece no computador de quem usa.

## Duas formas de usar
- **No site (GitHub Pages):** abre direto no navegador, sem instalar nada. O corretor inteiro roda dentro da
  página (Python via Pyodide) e os cartões e notas ficam guardados só naquele navegador. Na primeira vez baixa
  cerca de 40 MB. O nome e a turma são lidos com um OCR mais simples (Tesseract), então confira mais.
- **No Mac (`iniciar.command`):** mais rápido em lotes grandes (lê 5 cartões por vez em paralelo) e usa o OCR do
  macOS, que erra menos nomes.

## Usar
1. Abra o site, ou dê dois cliques em `iniciar.command` (abre http://127.0.0.1:5055).
2. **Gabarito**: envie os cartões "Gabarito de correção" (um por tipo de prova) ou digite. Ajuste quanto vale cada tipo de item.
3. **Cartões**: arraste os PDFs escaneados ou imagens.
4. **Resultados**: clique num aluno para conferir a imagem e corrigir a leitura; baixe a planilha.

## Como lê
- Endireita a folha pelos 4 marcadores (aceita folha torta ou de ponta-cabeça).
- Descobre as questões pelas bolhas: fileira com 2 = certo/errado, 4 ou 5 = múltipla escolha, bloco de 10 fileiras sem número = numérica.
- A numeração é pela posição (coluna da esquerda para a direita, de cima para baixo), não pelo número impresso.
- Nome, matrícula e tipo de prova vêm do OCR do macOS.

## Testes
`.venv/bin/python -m testes.test_leitor` — cartões sintéticos, sem dados de alunos.

## Teste de carga
`.venv/bin/python -m testes.carga 300` — gera 300 cartões sintéticos (PDF + JPGs soltos), envia a um corretor
separado (porta 5057, pasta de dados própria) e confere tempo, memória e notas. Não mexe nos cartões reais.
