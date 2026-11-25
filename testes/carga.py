"""Teste de carga: gera um lote grande de cartões sintéticos, envia ao corretor e confere tempo, memória e notas.

    .venv/bin/python -m testes.carga [quantidade=300] [pasta_de_trabalho]

Sobe um corretor separado (porta 5057, dados numa pasta própria), então não mexe nos cartões reais.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

PORTA = 5057
URL = f"http://127.0.0.1:{PORTA}"


def _pagina(args):
    from testes.gerar_sintetico import escanear, gerar
    modelo, i, tipo, gab = args
    if gab:
        img, v = gerar(modelo, semente=9000 + ord(tipo), branco=0, dupla=0, nome="GABARITO DE CORREÇÃO DA AVALIAÇÃO",
                       tipo_prova=tipo)
    else:
        img, v = gerar(modelo, semente=i, nome=f"ALUNO NUMERO {i:04d} DE TESTE", matricula=str(20260000000 + i),
                       numero=i % 1000, codigo=i, tipo_prova=tipo)
    ok, buf = cv2.imencode(".jpg", escanear(img, semente=i, vira=(i % 17 == 0)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return i, tipo, gab, v, buf.tobytes()


def gerar_lote(pasta: Path, qtd: int, modelo="hum"):
    """Um PDF com 2 gabaritos (tipos A e B) + qtd alunos, e 40 JPGs soltos. Devolve as respostas verdadeiras."""
    import pymupdf
    tarefas = [(modelo, 0, "A", True), (modelo, 1, "B", True)] + [(modelo, i, "AB"[i % 2], False) for i in range(2, qtd + 2)]
    chaves, verdade = {}, {}
    doc = pymupdf.open()
    soltos = pasta / "soltos"
    soltos.mkdir(exist_ok=True)
    with ProcessPoolExecutor(6) as pool:
        for i, tipo, gab, v, jpg in pool.map(_pagina, tarefas, chunksize=8):
            if gab:
                chaves[tipo] = v
            else:
                verdade[str(20260000000 + i)] = (tipo, v)
            if not gab and i >= qtd + 2 - 40:
                (soltos / f"cartao_{i:04d}.jpg").write_bytes(jpg)
            else:
                p = doc.new_page(width=595, height=842)
                p.insert_image(p.rect, stream=jpg)
    doc.save(pasta / "lote.pdf")
    return chaves, verdade


def _json(caminho, metodo="GET"):
    with urllib.request.urlopen(urllib.request.Request(URL + caminho, method=metodo), timeout=60) as r:
        return json.loads(r.read())


def _memoria_mb(pid: int) -> float:
    """Memória do corretor e dos processos filhos, em MB."""
    saida = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    linhas = [tuple(map(int, l.split())) for l in saida.splitlines() if l.strip()]
    familia, mudou = {pid}, True
    while mudou:
        mudou = False
        for p, pp, _ in linhas:
            if pp in familia and p not in familia:
                familia.add(p)
                mudou = True
    return sum(rss for p, _, rss in linhas if p in familia) / 1024


def _enviar(arquivos: list[Path]):
    cmd = ["curl", "-s"]
    for a in arquivos:
        cmd += ["-F", f"arquivos=@{a}"]
    subprocess.run(cmd + [URL + "/api/cartoes"], check=True, capture_output=True)


def _esperar(pid: int):
    pico = 0.0
    while True:
        time.sleep(1)
        pico = max(pico, _memoria_mb(pid))
        p = _json("/api/estado")["progresso"]
        if not p["ativo"]:
            return pico, p["erro"]


def main():
    qtd = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    pasta = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.mkdtemp(prefix="carga_"))
    pasta.mkdir(parents=True, exist_ok=True)
    t = time.time()
    chaves, verdade = gerar_lote(pasta, qtd)
    print(f"lote gerado em {time.time() - t:.0f}s: {pasta / 'lote.pdf'} ({(pasta / 'lote.pdf').stat().st_size / 1e6:.0f} MB) + 40 JPGs")

    raiz = Path(__file__).resolve().parent.parent
    env = dict(os.environ, DADOS=str(pasta / "dados"), PORTA=str(PORTA))
    srv = subprocess.Popen([sys.executable, "app.py"], cwd=raiz, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                _json("/api/estado")
                break
            except OSError:
                time.sleep(0.25)
        t = time.time()
        _enviar([pasta / "lote.pdf"])
        pico1, erro = _esperar(srv.pid)
        t1 = time.time() - t
        n1 = len(_json("/api/estado")["cartoes"])
        print(f"PDF: {n1} cartões em {t1:.0f}s ({n1 / t1:.1f} por segundo), pico de memória {pico1:.0f} MB {erro}")
        t = time.time()
        _enviar(sorted((pasta / "soltos").glob("*.jpg")))
        pico2, erro = _esperar(srv.pid)
        t2 = time.time() - t
        t = time.time()
        e = _json("/api/estado")
        t_estado = time.time() - t
        print(f"JPGs soltos: {len(e['cartoes']) - n1} cartões em {t2:.0f}s, pico {pico2:.0f} MB {erro}")
        print(f"gabaritos reconhecidos: {e['gabarito']}; tela (/api/estado) responde em {t_estado * 1000:.0f} ms")

        errados, sem_nome, com_aviso, faltando = 0, 0, 0, set(verdade)
        for c in e["cartoes"]:
            mat = c["info"].get("matricula", "")
            if mat not in verdade:
                sem_nome += 1
                continue
            faltando.discard(mat)
            tipo, v = verdade[mat]
            esperado = float(sum(1 for n, r in v.items() if r == chaves[tipo][n]))
            if c["info"].get("tipo_prova") != tipo or abs((c["nota"] or 0) - esperado) > 1e-6:
                errados += 1
            if [a for a in c["avisos"] if not a.startswith("dupla")]:
                com_aviso += 1
        print(f"notas: {len(verdade) - len(faltando) - errados} certas, {errados} erradas, "
              f"{len(faltando) + sem_nome} cartões sem matrícula lida, {com_aviso} com aviso além de dupla marcação")
        t = time.time()
        with urllib.request.urlopen(URL + "/exportar.xlsx", timeout=120) as r:
            tam = len(r.read())
        print(f"planilha: {tam / 1024:.0f} KB em {time.time() - t:.1f}s")
        return 1 if (errados or faltando or sem_nome or erro) else 0
    finally:
        srv.terminate()


if __name__ == "__main__":
    sys.exit(main())
