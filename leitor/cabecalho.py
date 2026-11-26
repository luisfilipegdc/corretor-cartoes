"""Lê a identificação do cartão (nome, matrícula, nº, tipo de prova, código) no cabeçalho.

Usa o OCR do próprio macOS (Vision). Se não estiver disponível, os campos ficam vazios
e a pessoa identifica o cartão pela imagem.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from . import omr


# letras cirílicas/gregas que o OCR troca pelas latinas de mesmo desenho
_PARECIDAS = str.maketrans("АВЕЗКМНОРСТХаеорсхΑΒΕΖΗΙΚΜΝΟΡΤΧ", "ABE3KMHOPCTXaeopcxABEZHIKMNOPTX")


def _ocr(img_cinza: np.ndarray) -> list[tuple[str, float, float]]:
    """[(texto, x_centro 0..1, y_topo 0..1)] de cima para baixo."""
    try:
        from ocrmac import ocrmac
        from PIL import Image
    except Exception:
        return []
    try:
        res = ocrmac.OCR(Image.fromarray(img_cinza), language_preference=["pt-BR"],
                         recognition_level="accurate").recognize()
    except Exception:
        return []
    linhas = [(t.strip(), b[0] + b[2] / 2, 1 - b[1] - b[3]) for t, _c, b in res if t.strip()]
    return sorted(linhas, key=lambda l: (round(l[2] * 40), l[1]))


def ler(reta: np.ndarray | None) -> dict:
    """Identificação lida do cabeçalho; com `None` devolve os campos vazios."""
    info = {"nome": "", "matricula": "", "numero": "", "tipo_prova": "", "codigo": "", "turma": "", "prova": "", "qr": "", "gabarito": False}
    if reta is None:
        return info
    topo = reta[: omr.MT - 22]
    info = interpretar(_ocr(topo))
    try:
        qr = topo[:, int(topo.shape[1] * 0.72):]
        qr = cv2.resize(qr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dado, _, _ = cv2.QRCodeDetector().detectAndDecode(qr)
        info["qr"] = (dado or "")[:200]
    except (cv2.error, AttributeError):
        pass
    return info


def interpretar(linhas: list[tuple[str, float, float]]) -> dict:
    """Extrai os campos das linhas do OCR: [(texto, x do centro 0..1, y do topo 0..1)], de cima para baixo."""
    info = {"nome": "", "matricula": "", "numero": "", "tipo_prova": "", "codigo": "", "turma": "", "prova": "", "qr": "", "gabarito": False}
    linhas = sorted(linhas, key=lambda l: (round(l[2] * 40), l[1]))
    texto = "\n".join(l[0] for l in linhas)
    info["gabarito"] = bool(re.search(r"(?i)GABARITO\s+DE\s+CORRE", texto))
    # a barra entre a matrícula e o nome às vezes é lida como "!", "l", "1"…: depois de "Matrícula/Nome:" aceita qualquer sinal
    m = (re.search(r"(?i)Matr[^\n:]{0,14}:\s*(\d{6,})[^\w\n]{0,4}(?:[lI1]\s+)?([^\n]+)", texto)
         or re.search(r"(\d{6,})\s*[/|!\\]\s*([^\n]+)", texto))
    if m:
        info["matricula"] = m.group(1)
        nome = re.sub(r"\s+TIPO\s+[A-Z]\s*$", "", m.group(2))
        limpo = re.match(r"[A-Za-zÀ-ÿ' .-]+", nome.strip())      # corta lixo que o OCR põe depois do nome
        info["nome"] = (limpo.group(0) if limpo else nome).strip(" -_/.")
    m = re.search(r"N[º°o\.]{0,2}\s*[:;]?\s*(\d{1,4})\s*-", texto)
    if m:
        info["numero"] = m.group(1)
    m = re.search(r"\bTIPO\s+([A-Z])\b", texto)
    if m:
        info["tipo_prova"] = m.group(1)
    m = re.search(r"\*+\s*(\d{4,})\s*\*+", texto) or re.search(r"(?m)^\D{0,3}(0\d{5})\D{0,3}$", texto)
    if m:
        info["codigo"] = m.group(1)
    # turma: linha do tipo "EM3AM - 2026" (o OCR às vezes devolve letras cirílicas parecidas)
    y_turma = None
    for t, x, y in linhas:
        m = re.fullmatch(r"\W*(\w{3,10})\s*[-–—]\s*(20\d\d)\W*", t.translate(_PARECIDAS).strip())
        if m and x < 0.5:
            info["turma"], y_turma = m.group(1).upper(), y
            break
    # nome da prova: linhas do quadro da esquerda, abaixo da turma (ou da escola) e acima do "Nº"
    y_fim = min([y for t, x, y in linhas if re.search(r"Matr|Assinatura", t)], default=0.62)
    y_ini = y_turma if y_turma is not None else min([y for t, x, y in linhas if x < 0.5], default=0) + 0.02
    titulo = [t for t, x, y in linhas if 0.12 < x < 0.5 and y_ini + 0.01 < y < y_fim - 0.03
              and not re.match(r"[°º•]", t)]
    info["prova"] = " ".join(titulo)[:160]
    # OCR mais fraco troca o dígito da série por letra parecida ("EM3AM" vira "EMSAM"): a série também está
    # no nome da prova ("3ª Série"), então usamos esse dígito
    t = info["turma"]
    serie = re.search(r"(\d)\s*[ªºa°]\s*S[ée]rie", info["prova"], re.I)
    if serie and len(t) >= 4 and not any(ch.isdigit() for ch in t) and t[2] in "SZBOIGE":
        info["turma"] = t[:2] + serie.group(1) + t[3:]
    return info
