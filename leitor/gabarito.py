"""Gabarito (texto, CSV ou planilha) e correção."""
from __future__ import annotations

import re

PESOS_PADRAO = {
    "CE": {"acerto": 1.0, "erro": 0.0},
    "NUM": {"acerto": 1.0, "erro": 0.0},
    "ABCDE": {"acerto": 1.0, "erro": 0.0},
}
ANULADA = "X"


def _grupo(tipo: str) -> str:
    return "ABCDE" if tipo.startswith("AB") else tipo


def _norm(resp: str) -> str:
    r = str(resp).strip().upper()
    if r in ("X", "ANULADA", "NULA", "ANUL", "*"):
        return ANULADA
    if r in ("CERTO", "CERTA", "V"):
        return "C"
    if r in ("ERRADO", "ERRADA", "F"):
        return "E"
    if re.fullmatch(r"\d+(\.0)?", r):
        return r.split(".")[0]
    return r


def ler_texto(texto: str) -> tuple[dict[str, dict[int, str]], list[str]]:
    """Aceita linhas como `1 C`, `19: 011`, `21-D`, `22 X` (anulada), várias por linha separadas por vírgula.

    Uma linha `TIPO B` abre o gabarito daquele tipo de prova; sem essa linha vale para todos ('*').
    Devolve ({tipo: {questão: resposta}}, avisos).
    """
    gab: dict[str, dict[int, str]] = {}
    avisos = []
    atual = "*"
    for bruta in texto.replace("\r", "").split("\n"):
        linha = bruta.strip()
        if not linha or linha.startswith("#"):
            continue
        m = re.fullmatch(r"(?i)tipo\s*[:\-]?\s*([A-Z0-9])\s*:?", linha)
        if m:
            atual = m.group(1).upper()
            continue
        pares = re.findall(r"(\d{1,3})\s*[\s:=\-\.\)\t;]\s*([A-Za-z]+(?:\+[A-Za-z])*|\d+|\*)(?=\s*(?:[,;|]|$|\s))", linha)
        if not pares:
            avisos.append(f"não entendi a linha: {linha!r}")
            continue
        for n, r in pares:
            gab.setdefault(atual, {})[int(n)] = _norm(r)
    return gab, avisos


def ler_planilha(caminho: str) -> tuple[dict[str, dict[int, str]], list[str]]:
    """Planilha .xlsx/.csv: 1ª coluna = questão; 2ª = resposta. Colunas extras com título 'Tipo B' etc. = outros tipos."""
    linhas: list[list] = []
    if caminho.lower().endswith(".xlsx"):
        import openpyxl
        ws = openpyxl.load_workbook(caminho, data_only=True).active
        linhas = [list(l) for l in ws.iter_rows(values_only=True)]
    else:
        import csv
        with open(caminho, newline="", encoding="utf-8-sig") as f:
            amostra = f.read()
        sep = ";" if amostra.count(";") > amostra.count(",") else ","
        linhas = [l for l in csv.reader(amostra.splitlines(), delimiter=sep)]
    gab: dict[str, dict[int, str]] = {}
    tipos = ["*"]
    for l in linhas:
        cel = [("" if c is None else str(c).strip()) for c in l]
        if not any(cel):
            continue
        if not re.fullmatch(r"\d+(\.0)?", cel[0]):
            tipos = []
            for c in cel[1:]:
                m = re.search(r"(?i)tipo\s*([A-Z0-9])\b", c)
                tipos.append(m.group(1).upper() if m else "*")
            continue
        n = int(float(cel[0]))
        for t, c in zip(tipos, cel[1:]):
            if c:
                gab.setdefault(t, {})[n] = _norm(c)
    return gab, ([] if gab else ["não achei nenhuma questão na planilha"])


def para_texto(gab: dict[str, dict[int, str]]) -> str:
    partes = []
    for t in sorted(gab, key=lambda k: (k != "*", k)):
        if t != "*":
            partes.append(f"TIPO {t}")
        partes += [f"{n} {r}" for n, r in sorted(gab[t].items())]
        partes.append("")
    return "\n".join(partes).strip()


def escolher(gab: dict[str, dict[int, str]], tipo_prova: str) -> dict[int, str] | None:
    if tipo_prova and tipo_prova in gab:
        return gab[tipo_prova]
    if "*" in gab:
        return gab["*"]
    if len(gab) == 1:
        return next(iter(gab.values()))
    return None


def corrigir(respostas: list[dict], chave: dict[int, str] | None, pesos: dict) -> dict:
    """Compara as respostas lidas com o gabarito. status: certa, errada, branco, dupla, anulada, sem_gabarito."""
    itens, nota, maximo = [], 0.0, 0.0
    cont = {"certa": 0, "errada": 0, "branco": 0, "dupla": 0, "anulada": 0, "sem_gabarito": 0}
    for r in respostas:
        g = (chave or {}).get(r["n"], "")
        p = pesos.get(_grupo(r["tipo"]), {"acerto": 1.0, "erro": 0.0})
        lido = r["resposta"]
        if r["tipo"] == "NUM" and g.isdigit():
            g = g.zfill(len(lido) if lido else 0) if lido else g
        pontos = 0.0
        if not g:
            st = "sem_gabarito"
        elif g == ANULADA:
            st, pontos = "anulada", p["acerto"]
            maximo += p["acerto"]
        else:
            maximo += p["acerto"]
            if not lido:
                st = "branco"
            elif "+" in g and lido in g.split("+"):   # gabarito com mais de uma alternativa aceita
                st, pontos = "certa", p["acerto"]
            elif r.get("dupla") or "+" in lido or "?" in lido:
                st, pontos = "dupla", -abs(p["erro"])
            elif lido == g:
                st, pontos = "certa", p["acerto"]
            else:
                st, pontos = "errada", -abs(p["erro"])
        cont[st] += 1
        nota += pontos
        itens.append({"n": r["n"], "tipo": r["tipo"], "lido": lido, "gabarito": g, "status": st, "pontos": pontos})
    return {"itens": itens, "nota": round(nota, 3), "maximo": round(maximo, 3), **cont}


def trocar_bloco(texto: str, tipo_prova: str, respostas: dict[int, str]) -> str:
    """Devolve o texto do gabarito com o bloco daquele tipo de prova substituído."""
    gab, _ = ler_texto(texto)
    gab[tipo_prova or "*"] = dict(respostas)
    if tipo_prova and "*" in gab:   # gabarito por tipo substitui o genérico
        del gab["*"]
    return para_texto(gab)
