"""Corretor de cartões resposta — página local. Rode com `./iniciar.command` e abra http://127.0.0.1:5055"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import uuid
from collections import Counter
from pathlib import Path

import cv2
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from leitor import cabecalho, gabarito, lote, omr

RAIZ = Path(__file__).parent
DADOS = Path(os.environ.get("DADOS") or RAIZ / "dados")
NAVEGADOR = sys.platform == "emscripten"   # rodando dentro do navegador (Pyodide)
POR_VEZ = max(1, int(os.environ.get("POR_VEZ", 5)))   # quantos cartões são lidos ao mesmo tempo
RETAS = DADOS / "retas"
ARQ_ESTADO = DADOS / "estado.json"
RETAS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=None)
trava = threading.RLock()
progresso = {"ativo": False, "feito": 0, "total": 0, "etapa": "", "erro": "", "gabaritos": []}


def _novo_estado():
    return {"gabarito_texto": "", "pesos": json.loads(json.dumps(gabarito.PESOS_PADRAO)), "cartoes": [], "layouts": {}}


def _carregar():
    if ARQ_ESTADO.exists():
        try:
            e = json.loads(ARQ_ESTADO.read_text(encoding="utf-8"))
            base = _novo_estado()
            base.update(e)
            return base
        except json.JSONDecodeError:
            pass
    return _novo_estado()


estado = _carregar()


def _salvar():
    tmp = ARQ_ESTADO.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ARQ_ESTADO)


# ---------------------------------------------------------------- processamento

def _fechar_lote(lidos: list[dict]) -> list[dict]:
    """Escolhe o layout de referência, lê as marcações e separa os cartões-gabarito. Devolve os de alunos."""
    bons = [it for it in lidos if it["lay"] is not None]
    with trava:
        def encaixe(it):
            """Layout já conhecido em que as bolhas deste cartão se encaixam (assinatura, aderência)."""
            cands = [(omr.aderencia(l, it["bolhas"]), s) for s, l in estado["layouts"].items()
                     if abs(l["H"] - it["lay"]["H"]) < 0.02 * l["H"]]
            a, s = max(cands, default=(0, ""))
            return (s, a) if a >= 0.9 else ("", a)

        # layout novo só é criado quando o cartão não se encaixa em nenhum já conhecido: assim um cartão
        # mal lido é corrigido pelo layout dos anteriores em vez de virar uma "prova nova"
        novos = [it for it in bons if omr.assinatura(it["lay"]) not in estado["layouts"] and not encaixe(it)[0]]
        cont = Counter(omr.assinatura(it["lay"]) for it in novos)
        minimo = 1 if len(bons) <= 2 else 2
        for sig, qtd in cont.items():
            limpos = [it["lay"] for it in novos if omr.assinatura(it["lay"]) == sig and not it["lay"]["avisos"]]
            if qtd >= minimo and limpos:
                estado["layouts"][sig] = omr.media_layouts(limpos)
        for it in lidos:
            c = it["c"]
            if it["lay"] is None:
                continue
            sig = omr.assinatura(it["lay"])
            ref = estado["layouts"].get(sig)
            if ref is None:
                melhor = encaixe(it)[0]
                if melhor:
                    sig, ref = melhor, estado["layouts"][melhor]
                    c["avisos"].append("leitura ajustada pelo layout dos outros cartões — confira")
                else:
                    estado["layouts"][sig] = it["lay"]
                    ref = it["lay"]
                    c["avisos"].append("layout diferente dos demais cartões — confira a imagem")
                    c["avisos"] += it["lay"]["avisos"]
            dx, dy = omr.deslocamento(ref, it["bolhas"])
            reta = it["reta"] if it["reta"] is not None else cv2.imread(str(RETAS / f"{c['id']}.jpg"),
                                                                        cv2.IMREAD_GRAYSCALE)
            resp = omr.ler(reta, ref, dx, dy)
            c["layout"], c["dx"], c["dy"] = sig, dx, dy
            c["respostas"] = [{k: v for k, v in r.items() if k != "fills"} for r in resp]
            it["reta"] = None
        # cartões "GABARITO DE CORREÇÃO" viram gabarito, não aluno
        for it in lidos:
            c = it["c"]
            if c["info"].get("gabarito") and c["respostas"]:
                tipo = c["info"].get("tipo_prova", "")
                estado["gabarito_texto"] = gabarito.trocar_bloco(
                    estado["gabarito_texto"], tipo, {r["n"]: r["resposta"] for r in c["respostas"]})
                ruins = [str(r["n"]) for r in c["respostas"] if not r["resposta"] or r["duvida"] or "?" in r["resposta"]]
                duplas = [str(r["n"]) for r in c["respostas"] if "+" in r["resposta"]]
                progresso["gabaritos"].append(("Tipo " + tipo if tipo else "único")
                                              + (" (confira as questões " + ", ".join(ruins) + ")" if ruins else "")
                                              + (" (aceitam mais de uma resposta: " + ", ".join(duplas) + ")" if duplas else ""))
                (RETAS / f"{c['id']}.jpg").unlink(missing_ok=True)
        lidos = [it for it in lidos if not (it["c"]["info"].get("gabarito") and it["c"]["respostas"])]
        return [it["c"] for it in lidos]


def _reler():
    """Refaz a leitura dos cartões já enviados (a partir das imagens guardadas), mantendo os editados à mão."""
    try:
        with trava:
            antigos = list(estado["cartoes"])
        progresso.update(etapa="Relendo cartões", feito=0, total=len(antigos))
        lidos, mantidos = [], []
        for c in antigos:
            reta = cv2.imread(str(RETAS / f"{c['id']}.jpg"), cv2.IMREAD_GRAYSCALE)
            if reta is None:
                mantidos.append(c)
                progresso["feito"] += 1
                continue
            H = reta.shape[0] - omr.MT - omr.MB
            bolhas = omr.achar_bolhas(reta, H)
            lay = omr.montar_layout(reta, bolhas, H)
            novo = cabecalho.ler(reta)
            for k, v in novo.items():   # completa o que faltava; o que já existe (ou foi digitado) fica
                if v and (k == "prova" or (k == "turma" and not c["editado"])):
                    c["info"][k] = v
                elif not c["info"].get(k):
                    c["info"][k] = v
            # correção manual só é mantida se as questões continuam as mesmas
            if c["editado"] and lay is not None and omr.assinatura(lay) == c.get("layout"):
                mantidos.append(c)
            else:
                era_editado = c["editado"]
                c.update(respostas=[], avisos=[], erro="", layout="", editado=False)
                if era_editado:
                    c["avisos"].append("relido: as correções manuais anteriores foram descartadas porque as questões mudaram")
                if lay is None:
                    c["erro"] = "não achei as bolhas de resposta"
                lidos.append({"c": c, "reta": reta, "bolhas": bolhas, "lay": lay})
            progresso["feito"] += 1
        with trava:
            usados = {c.get("layout") for c in mantidos}
            estado["layouts"] = {k: v for k, v in estado["layouts"].items() if k in usados}
            estado["cartoes"] = mantidos + _fechar_lote(lidos)
            _salvar()
    except Exception as ex:
        progresso["erro"] = f"{type(ex).__name__}: {ex}"
    finally:
        progresso.update(ativo=False, etapa="")


def _processar(arquivos: list[tuple[str, str]]):
    """arquivos: [(nome original, caminho temporário)].

    Lê de POR_VEZ em POR_VEZ páginas: cada grupo é lido em paralelo, identificado e já entra na lista de
    resultados, então os cartões vão aparecendo na tela e a memória usada não cresce com o tamanho do lote.
    """
    nomes = lote.LeitorCabecalhos()
    pool = None
    try:
        tarefas = []
        progresso.update(etapa="Abrindo arquivos", feito=0, total=len(arquivos))
        for nome, caminho in arquivos:
            try:
                tarefas += [(caminho, i, nome) for i in range(lote.contar_paginas(caminho))]
            except Exception:
                tarefas.append((caminho, 0, nome))   # ilegível: vira um cartão com erro na lista
            progresso["feito"] += 1
        progresso.update(etapa="Lendo cartões", feito=0, total=len(tarefas))
        if len(tarefas) > 2:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=min(POR_VEZ, len(tarefas)))
        for ini in range(0, len(tarefas), POR_VEZ):
            grupo = tarefas[ini:ini + POR_VEZ]
            if pool is None:
                lidos = [lote.ler_pagina(cam, i, nome, str(RETAS)) for cam, i, nome in grupo]
            else:
                futuros = [pool.submit(lote.ler_pagina, cam, i, nome, str(RETAS)) for cam, i, nome in grupo]
                lidos = [f.result() for f in futuros]   # na ordem das páginas
            com_imagem = {str(RETAS / f"{it['c']['id']}.jpg"): it["c"] for it in lidos if it["bolhas"] is not None}
            infos = nomes.ler(list(com_imagem))
            for caminho, c in com_imagem.items():
                c["info"] = infos.get(caminho) or cabecalho.ler(None)
            with trava:
                estado["cartoes"] += _fechar_lote(lidos)
                _salvar()
            progresso["feito"] += len(grupo)
    except Exception as ex:
        progresso["erro"] = f"{type(ex).__name__}: {ex}"
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        nomes.fechar()
        for _n, caminho in arquivos:
            try:
                os.unlink(caminho)
            except OSError:
                pass
        progresso.update(ativo=False, etapa="")


# ---------------------------------------------------------------- correção

def _gabaritos():
    return gabarito.ler_texto(estado["gabarito_texto"])


def _corrigido(c: dict, gab: dict) -> dict | None:
    if c["erro"] or not c["respostas"]:
        return None
    chave = gabarito.escolher(gab, c["info"].get("tipo_prova", ""))
    return gabarito.corrigir(c["respostas"], chave, estado["pesos"])


def _avisos(c: dict, cor: dict | None, gab: dict) -> list[str]:
    av = list(c["avisos"])
    if c["erro"]:
        return [c["erro"]]
    duv = [str(r["n"]) for r in c["respostas"] if r.get("duvida")]
    if duv:
        av.append("marcação fraca ou incompleta: " + ", ".join(duv))
    if cor and cor["dupla"]:
        av.append("dupla marcação: " + ", ".join(str(i["n"]) for i in cor["itens"] if i["status"] == "dupla"))
    if not c["info"].get("nome"):
        av.append("nome não lido")
    if gab:
        chave = gabarito.escolher(gab, c["info"].get("tipo_prova", ""))
        if chave is None:
            tipo, sug = c["info"].get("tipo_prova", ""), _sugerir_tipo(c, gab)
            motivo = f"não há gabarito do tipo {tipo}" if tipo else "tipo de prova não lido"
            av.append(f"não sei qual gabarito usar ({motivo})" + (
                f" — parece Tipo {sug['tipo']}: {sug['certas']} acertos contra {sug['outro']}; confirme" if sug else ""))
        elif cor:
            ns = {r["n"] for r in c["respostas"]}
            if ns != set(chave):
                av.append(f"cartão tem {len(ns)} questões e o gabarito tem {len(chave)}")
            ruins = [str(r["n"]) for r in c["respostas"] if r["n"] in chave and chave[r["n"]] != gabarito.ANULADA
                     and (chave[r["n"]].isdigit() != (r["tipo"] == "NUM")
                          or (r["tipo"] != "NUM" and not set(chave[r["n"]].split("+")) <= set(_opcoes(r["tipo"]))))]
            if ruins:
                av.append("gabarito não combina com o tipo da questão: " + ", ".join(ruins[:12]))
    return av


def _sugerir_tipo(c: dict, gab: dict) -> dict | None:
    """Cartão sem gabarito definido: compara com o gabarito de cada tipo e sugere o que combina bem mais."""
    if c["erro"] or not c["respostas"] or gabarito.escolher(gab, c["info"].get("tipo_prova", "")) is not None:
        return None
    placar = sorted(((gabarito.corrigir(c["respostas"], chave, estado["pesos"])["certa"], t)
                     for t, chave in gab.items() if t != "*"), reverse=True)
    n = len(c["respostas"])   # só sugere com folga: aluno de nota baixa combina mal com os dois gabaritos
    if len(placar) < 2 or placar[0][0] < n / 2 or placar[0][0] - placar[1][0] < max(3, round(0.1 * n)):
        return None
    return {"tipo": placar[0][1], "certas": placar[0][0], "outro": placar[1][0]}


def _opcoes(tipo: str) -> list[str]:
    return {"CE": ["C", "E"]}.get(tipo, list(tipo))


def _resumo(c: dict, gab: dict) -> dict:
    cor = _corrigido(c, gab)
    return {"id": c["id"], "arquivo": c["arquivo"], "pagina": c["pagina"], "info": c["info"], "erro": c["erro"],
            "editado": c["editado"], "avisos": _avisos(c, cor, gab), "tipo_sugerido": _sugerir_tipo(c, gab), "questoes": len(c["respostas"]),
            "nota": cor and cor["nota"], "maximo": cor and cor["maximo"],
            "certa": cor and cor["certa"], "errada": cor and cor["errada"],
            "branco": cor and cor["branco"], "dupla": cor and cor["dupla"]}


def _resumos(gab: dict) -> list[dict]:
    """Resumo de todos os cartões, avisando quando o mesmo aluno aparece mais de uma vez."""
    chave = lambda c: (c["info"].get("matricula") or c["info"].get("codigo") or "")
    vezes = Counter(chave(c) for c in estado["cartoes"] if chave(c))
    saida = []
    for c in estado["cartoes"]:
        r = _resumo(c, gab)
        if vezes.get(chave(c), 0) > 1:
            r["avisos"].append(f"aluno aparece {vezes[chave(c)]} vezes — exclua o repetido")
        saida.append(r)
    return saida


def _achar(cid: str) -> dict:
    for c in estado["cartoes"]:
        if c["id"] == cid:
            return c
    abort(404)


# ---------------------------------------------------------------- rotas

@app.get("/")
def pagina():
    return send_from_directory(RAIZ, "index.html")


@app.get("/api/estado")
def api_estado():
    with trava:
        gab, avisos = _gabaritos()
        return jsonify({
            "gabarito_texto": estado["gabarito_texto"], "pesos": estado["pesos"],
            "gabarito": {t: len(q) for t, q in gab.items()}, "gabarito_avisos": avisos,
            "cartoes": _resumos(gab), "progresso": progresso,
        })


@app.post("/api/gabarito")
def api_gabarito():
    d = request.get_json(force=True)
    with trava:
        estado["gabarito_texto"] = str(d.get("texto", ""))
        for tipo, p in (d.get("pesos") or {}).items():
            if tipo in estado["pesos"]:
                estado["pesos"][tipo] = {"acerto": float(p.get("acerto", 1)), "erro": abs(float(p.get("erro", 0)))}
        _salvar()
    return api_estado()


@app.delete("/api/gabarito/<tipo>")
def api_gabarito_remover(tipo):
    """Tira o gabarito de um tipo de prova ('*' = o gabarito sem tipo)."""
    with trava:
        gab, _ = gabarito.ler_texto(estado["gabarito_texto"])
        gab.pop(tipo, None)
        estado["gabarito_texto"] = gabarito.para_texto(gab)
        _salvar()
    return api_estado()


@app.post("/api/gabarito-arquivo")
def api_gabarito_arquivo():
    """Planilha/CSV/TXT com o gabarito, ou um cartão preenchido com as respostas certas. Devolve o texto."""
    f = request.files.get("arquivo")
    nome = f.filename if f else request.args.get("nome", "")   # no navegador o arquivo vem como corpo puro
    if not f and not request.get_data():
        abort(400)
    ext = os.path.splitext(nome or "")[1].lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        if f:
            f.save(tmp)
        else:
            tmp.write(request.get_data())
        caminho = tmp.name
    try:
        if ext in (".xlsx", ".csv"):
            gab, avisos = gabarito.ler_planilha(caminho)
            return jsonify({"texto": gabarito.para_texto(gab), "avisos": avisos})
        if ext == ".txt":
            return jsonify({"texto": Path(caminho).read_text(encoding="utf-8-sig"), "avisos": []})
        texto, avisos = estado["gabarito_texto"] if (request.form.get("somar") or request.args.get("somar")) else "", []
        for img in omr.carregar_paginas(caminho):
            reta, H = omr.endireitar(img)
            lay = omr.montar_layout(reta, omr.achar_bolhas(reta, H), H)
            if lay is None:
                raise omr.ErroLeitura("não achei as bolhas de resposta")
            resp = omr.ler(reta, lay)
            tipo = cabecalho.ler(reta).get("tipo_prova", "")
            ruins = [str(r["n"]) for r in resp if not r["resposta"] or r["duvida"] or "?" in r["resposta"]]
            duplas = [str(r["n"]) for r in resp if "+" in r["resposta"]]
            if ruins:
                avisos.append((f"Tipo {tipo}: " if tipo else "") + "confira as questões (em branco ou marcação fraca) "
                              + ", ".join(ruins))
            if duplas:
                avisos.append((f"Tipo {tipo}: " if tipo else "") + "duas alternativas marcadas (qualquer uma será aceita): "
                              + ", ".join(duplas))
            avisos += lay["avisos"]
            texto = gabarito.trocar_bloco(texto, tipo, {r["n"]: r["resposta"] for r in resp})
        return jsonify({"texto": texto, "avisos": avisos})
    except omr.ErroLeitura as ex:
        return jsonify({"erro": str(ex)}), 400
    finally:
        os.unlink(caminho)


@app.post("/api/cartoes")
def api_cartoes():
    if progresso["ativo"]:
        return jsonify({"erro": "já há uma leitura em andamento"}), 409
    arquivos = []
    for f in request.files.getlist("arquivos"):
        ext = os.path.splitext(f.filename or "")[1].lower()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp)
            arquivos.append((f.filename, tmp.name))
    if not arquivos:
        abort(400)
    progresso.update(ativo=True, feito=0, total=len(arquivos), etapa="Abrindo arquivos", erro="", gabaritos=[])
    threading.Thread(target=_processar, args=(arquivos,), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/reler")
def api_reler():
    if progresso["ativo"]:
        return jsonify({"erro": "já há uma leitura em andamento"}), 409
    progresso.update(ativo=True, feito=0, total=len(estado["cartoes"]), etapa="Relendo cartões", erro="", gabaritos=[])
    if NAVEGADOR:
        _reler()
    else:
        threading.Thread(target=_reler, daemon=True).start()
    return jsonify({"ok": True})


# ---- rotas usadas só quando o corretor roda dentro do navegador (GitHub Pages): lá não há processos nem
# ---- OCR do macOS, então a página manda uma folha por vez e o texto do cabeçalho já lido

pendentes: dict[str, dict] = {}


@app.post("/api/pagina")
def api_pagina():
    """Corpo: bytes de uma imagem (uma folha). Lê as bolhas e deixa o cartão pendente até /api/grupo."""
    import numpy as np
    dados = np.frombuffer(request.get_data(), dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR) if len(dados) else None
    item = lote.ler_imagem(img, int(request.args.get("pagina", 1)) - 1, request.args.get("nome", "cartão"), str(RETAS))
    item["c"]["info"] = cabecalho.ler(None)
    pendentes[item["c"]["id"]] = item
    return jsonify({"id": item["c"]["id"], "erro": item["c"]["erro"], "tem_imagem": item["bolhas"] is not None})


@app.get("/cabecalho/<cid>.png")
def cabecalho_png(cid):
    reta = cv2.imread(str(RETAS / f"{cid}.jpg"), cv2.IMREAD_GRAYSCALE)
    if reta is None:
        abort(404)
    topo = reta[: omr.MT - 22].copy()
    h, w = topo.shape
    topo[: int(h * 0.845), int(w * 0.835):] = 255      # apaga o QR code, que o OCR do navegador lê como lixo
    topo[: int(h * 0.60), int(w * 0.455): int(w * 0.835)] = 255   # e o quadro de instruções, que embaralha as linhas
    topo = cv2.resize(topo, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", topo)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png", max_age=0)


@app.post("/api/pagina/<cid>/texto")
def api_pagina_texto(cid):
    """Corpo: {"linhas": [[texto, x_centro, y_topo], ...]} vindas do OCR feito na página."""
    if cid not in pendentes:
        abort(404)
    linhas = [(str(t), float(x), float(y)) for t, x, y in request.get_json(force=True).get("linhas", [])]
    pendentes[cid]["c"]["info"] = cabecalho.interpretar(linhas)
    return jsonify(pendentes[cid]["c"]["info"])


@app.post("/api/grupo")
def api_grupo():
    """Fecha o grupo de cartões pendentes: escolhe o layout, lê as marcações e põe na lista."""
    progresso["gabaritos"] = []
    with trava:
        lidos = list(pendentes.values())
        pendentes.clear()
        estado["cartoes"] += _fechar_lote(lidos)
        _salvar()
    return jsonify({"gabaritos": progresso["gabaritos"]})


@app.delete("/api/cartoes")
def api_limpar():
    with trava:
        if progresso["ativo"]:
            return jsonify({"erro": "leitura em andamento"}), 409
        for c in estado["cartoes"]:
            (RETAS / f"{c['id']}.jpg").unlink(missing_ok=True)
        estado["cartoes"], estado["layouts"] = [], {}
        _salvar()
    return api_estado()


@app.get("/api/cartao/<cid>")
def api_cartao(cid):
    with trava:
        c = _achar(cid)
        gab, _ = _gabaritos()
        cor = _corrigido(c, gab)
        return jsonify({**_resumo(c, gab), "itens": cor["itens"] if cor else [],
                        "duvidas": [r["n"] for r in c["respostas"] if r.get("duvida")]})


@app.post("/api/cartao/<cid>")
def api_cartao_editar(cid):
    d = request.get_json(force=True)
    with trava:
        c = _achar(cid)
        for k in ("nome", "matricula", "numero", "tipo_prova", "turma"):
            if k in (d.get("info") or {}):
                c["info"][k] = str(d["info"][k]).strip().upper() if k == "tipo_prova" else str(d["info"][k]).strip()
        novas = {int(k): str(v).strip().upper() for k, v in (d.get("respostas") or {}).items()}
        for r in c["respostas"]:
            if r["n"] in novas and novas[r["n"]] != r["resposta"]:
                r["resposta"] = novas[r["n"]]
                r["dupla"] = "+" in r["resposta"]
                r["duvida"] = False
        c["editado"] = True
        _salvar()
    return api_cartao(cid)


@app.delete("/api/cartao/<cid>")
def api_cartao_apagar(cid):
    with trava:
        c = _achar(cid)
        estado["cartoes"].remove(c)
        (RETAS / f"{cid}.jpg").unlink(missing_ok=True)
        _salvar()
    return jsonify({"ok": True})


@app.get("/img/<cid>.jpg")
def imagem(cid):
    with trava:
        c = _achar(cid)
        gab, _ = _gabaritos()
        cor = _corrigido(c, gab)
        lay = estado["layouts"].get(c.get("layout", ""))
    reta = cv2.imread(str(RETAS / f"{cid}.jpg"), cv2.IMREAD_GRAYSCALE)
    if reta is None:
        abort(404)
    if lay and c["respostas"]:
        tem_gab = cor and any(i["status"] != "sem_gabarito" for i in cor["itens"])
        img = omr.conferencia(reta, lay, c["respostas"], cor["itens"] if tem_gab else None,
                              c.get("dx", 0), c.get("dy", 0))
    else:
        img = reta
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg", max_age=0)


@app.get("/exportar.xlsx")
def exportar():
    import openpyxl
    from openpyxl.styles import Font, PatternFill
    with trava:
        gab, _ = _gabaritos()
        linhas = [(c, _corrigido(c, gab), _avisos(c, _corrigido(c, gab), gab)) for c in estado["cartoes"]]
    linhas.sort(key=lambda l: (l[0]["info"].get("turma", ""), l[0]["info"].get("nome", "")))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Notas"
    ws.append(["Nº", "Matrícula", "Nome", "Turma", "Tipo", "Nota", "Máximo", "%", "Acertos", "Erros", "Em branco",
               "Dupla marcação", "Avisos", "Prova", "Arquivo", "Página"])
    for c, cor, av in linhas:
        i = c["info"]
        pct = round(100 * cor["nota"] / cor["maximo"], 1) if cor and cor["maximo"] else None
        ws.append([i.get("numero", ""), i.get("matricula", ""), i.get("nome", ""), i.get("turma", ""),
                   i.get("tipo_prova", ""), cor and cor["nota"], cor and cor["maximo"], pct,
                   cor and cor["certa"], cor and cor["errada"], cor and cor["branco"], cor and cor["dupla"],
                   "; ".join(av), i.get("prova", ""), c["arquivo"], c["pagina"]])
    ns = sorted({r["n"] for c, _, _ in linhas for r in c["respostas"]})
    wr = wb.create_sheet("Respostas")
    wr.append(["Turma", "Matrícula", "Nome", "Tipo"] + ns)
    verde, verm = PatternFill("solid", fgColor="C6EFCE"), PatternFill("solid", fgColor="FFC7CE")
    acertos = Counter()
    validos = Counter()
    for c, cor, _ in linhas:
        if not c["respostas"]:
            continue
        por_n = {i["n"]: i for i in (cor["itens"] if cor else [])}
        lidas = {r["n"]: r["resposta"] for r in c["respostas"]}
        wr.append([c["info"].get("turma", ""), c["info"].get("matricula", ""), c["info"].get("nome", ""),
                   c["info"].get("tipo_prova", "")]
                  + [lidas.get(n, "") for n in ns])
        for col, n in enumerate(ns, 5):
            st = por_n.get(n, {}).get("status")
            if st in ("certa", "anulada"):
                wr.cell(wr.max_row, col).fill = verde
                acertos[n] += 1
            elif st in ("errada", "dupla"):
                wr.cell(wr.max_row, col).fill = verm
            if st and st != "sem_gabarito":
                validos[n] += 1
    wq = wb.create_sheet("Questões")
    wq.append(["Questão", "Cartões", "Acertos", "% de acerto"])
    for n in ns:
        wq.append([n, validos[n], acertos[n], round(100 * acertos[n] / validos[n], 1) if validos[n] else None])
    for aba in wb:
        for cel in aba[1]:
            cel.font = Font(bold=True)
        aba.freeze_panes = "A2"
    ws.column_dimensions["C"].width = 42
    ws.column_dimensions["M"].width = 60
    wr.column_dimensions["C"].width = 42
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="notas.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORTA", 5055)), debug=False, threaded=True)
