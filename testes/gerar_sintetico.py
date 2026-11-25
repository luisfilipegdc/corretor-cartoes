"""Gera cartões sintéticos parecidos com o modelo real (para testar a leitura sem dados de alunos)."""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 1654 / 1414  # desenhamos em 200 dpi usando as medidas do escaneado de 1414 px


def _fonte(tam, negrito=False):
    for nome in (("Arial Bold.ttf" if negrito else "Arial.ttf"), "Helvetica.ttc"):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Supplemental/" + nome, int(tam * S))
        except OSError:
            try:
                return ImageFont.truetype("/System/Library/Fonts/" + nome, int(tam * S))
            except OSError:
                continue
    return ImageFont.load_default()


MODELOS = {
    # (tipo, primeira questão, quantidade, x da primeira bolha, y da primeira fileira, passo y)
    "mat": {"quadro": (75, 318, 1345, 1915), "blocos": [
        ("CE", 1, 18, 362, 477, 37.1), ("NUM", 19, 1, 570, 515, 37.1), ("NUM", 20, 1, 720, 515, 37.1),
        ("ABCDE", 21, 10, 1003, 477, 37.2)]},
    "hum": {"quadro": (45, 272, 1372, 1948), "blocos": [
        ("CE", 1, 30, 502, 380, 38.9), ("CE", 31, 10, 895, 380, 38.9), ("ABCDE", 41, 20, 895, 775, 38.9)]},
    "hum2": {"quadro": (45, 272, 1372, 1948), "blocos": [
        ("CE", 1, 15, 462, 380, 38.9), ("ABCDE", 16, 5, 462, 972, 38.9), ("CE", 21, 7, 462, 1170, 38.9),
        ("ABCDE", 28, 3, 462, 1446, 38.9), ("CE", 31, 7, 935, 380, 38.9), ("ABCDE", 38, 3, 935, 658, 38.9),
        ("CE", 41, 15, 935, 778, 38.9), ("ABCDE", 56, 5, 935, 1368, 38.9)]},
}
PASSO_X = 37.3
RAIO = 13


def gerar(modelo="mat", semente=0, nome="ALUNO TESTE DA SILVA", matricula="20260000409", numero=31,
          codigo=203, tipo_prova="B", branco=0.05, dupla=0.03, fraca=0.0):
    """Devolve (imagem BGR 200 dpi sem distorção, respostas verdadeiras {n: resposta})."""
    rnd = random.Random(semente)
    m = MODELOS[modelo]
    img = Image.new("L", (1654, 2339), 255)
    d = ImageDraw.Draw(img)
    P = lambda *v: tuple(int(round(a * S)) for a in v)
    qx0, qy0, qx1, qy1 = m["quadro"]
    # cabeçalho
    d.rectangle(P(qx0, qy0 - 208, qx0 + 635, qy0 - 85), outline=0, width=2)
    d.text(P(qx0 + 148, qy0 - 203), "COLÉGIO EXEMPLO DE TESTE - UNIDADE", font=_fonte(21, True), fill=0)
    d.text(P(qx0 + 148, qy0 - 178), "EM3AM - 2026", font=_fonte(18), fill=0)
    d.text(P(qx0 + 148, qy0 - 155), "2ª Avaliação Integrada de Teste", font=_fonte(20, True), fill=0)
    d.rectangle(P(qx0 + 642, qy0 - 208, qx0 + 1068, qy0 - 85), outline=0, width=2)
    d.text(P(qx0 + 650, qy0 - 203), "° Use apenas caneta esferográfica preta ou azul.", font=_fonte(16), fill=0)
    d.rectangle(P(qx0, qy0 - 78, qx0 + 1068, qy0 - 10), outline=0, width=2)
    d.text(P(qx0 + 6, qy0 - 72), f"Nº: {numero:03d} - Matrícula/Nome: {matricula} / {nome}", font=_fonte(20, True), fill=0)
    d.text(P(qx0 + 6, qy0 - 42), "Data: 12/11/2026 - Assinatura: ______________________", font=_fonte(19), fill=0)
    d.text(P(qx0 + 918, qy0 - 40), f"TIPO {tipo_prova}", font=_fonte(19, True), fill=0)
    d.rectangle(P(qx0 + 1075, qy0 - 208, qx1 + 2, qy0 - 10), outline=0, width=2)
    try:
        import qrcode
        qr = qrcode.make(f"{codigo:06d}", box_size=4, border=1).convert("L").resize(P(125, 125))
        img.paste(qr, P(qx0 + 1108, qy0 - 172))
    except ImportError:
        pass
    d.text(P(qx0 + 1132, qy0 - 38), f"**{codigo:06d}**", font=_fonte(17), fill=0)
    # quadro, código de barras e marcadores
    d.rectangle(P(qx0, qy0, qx1, qy1), outline=0, width=2)
    x = qx0 + 110
    while x < qx1 - 180:
        w = rnd.choice([2, 2, 4, 6])
        d.rectangle(P(x, qy0 + 6, x + w, qy0 + 30), fill=0)
        x += w + rnd.choice([3, 4, 6])
    for mx, my, inv in ((qx0 + 17, qy0 + 17, 0), (qx1 - 17, qy0 + 17, 1), (qx1 - 17, qy1 - 17, 0), (qx0 + 17, qy1 - 17, 1)):
        d.rectangle(P(mx - 12, my - 12, mx + 12, my + 12), outline=0, width=3)
        a, b = ((-12, -12, 0, 0), (0, 0, 12, 12)) if not inv else ((0, -12, 12, 0), (-12, 0, 0, 12))
        for r in (a, b):
            d.rectangle(P(mx + r[0], my + r[1], mx + r[2], my + r[3]), fill=0)
    # questões
    verdade = {}
    fn, fl = _fonte(15, True), _fonte(11)

    def bolha(cx, cy, letra, cheia, fraca_=False):
        d.ellipse(P(cx - RAIO, cy - RAIO, cx + RAIO, cy + RAIO), outline=0, width=2, fill=None)
        d.text(P(cx - 4, cy - 7), letra, font=fl, fill=140)
        if cheia:
            rr = RAIO - rnd.choice([0, 1, 2])
            ox, oy = rnd.uniform(-1.5, 1.5), rnd.uniform(-1.5, 1.5)
            d.ellipse(P(cx - rr + ox, cy - rr + oy, cx + rr + ox, cy + rr + oy), fill=110 if fraca_ else rnd.randint(15, 60))

    for tipo, q0, qtd, bx, by, py in m["blocos"]:
        if tipo == "NUM":
            d.rectangle(P(bx - 18, by - 55, bx + 2 * PASSO_X + 18, by - 20), fill=200)
            d.text(P(bx + PASSO_X - 9, by - 47), str(q0), font=fn, fill=0)
            resp = ""
            for c in range(3):
                dig = rnd.randint(0, 9)
                resp += str(dig)
                for k in range(10):
                    bolha(bx + c * PASSO_X, by + k * py, str(k), k == dig)
            verdade[q0] = resp
            continue
        letras = "CE" if tipo == "CE" else tipo
        for i in range(qtd):
            y = by + i * py
            if i % 2:
                d.rectangle(P(bx - 18, y - py / 2, bx + (len(letras) - 1) * PASSO_X + 18, y + py / 2), fill=232)
            d.rectangle(P(bx - 56, y - py / 2, bx - 19, y + py / 2), fill=185)
            d.text(P(bx - 50 + (8 if q0 + i < 10 else 0), y - 9), str(q0 + i), font=fn, fill=0)
            sorte = rnd.random()
            if sorte < branco:
                marc = []
            elif sorte < branco + dupla:
                marc = rnd.sample(letras, 2)
            else:
                marc = [rnd.choice(letras)]
            for j, l in enumerate(letras):
                bolha(bx + j * PASSO_X, y, l, l in marc, fraca_=rnd.random() < fraca)
            verdade[q0 + i] = "+".join(sorted(marc, key=letras.index))
    return cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), verdade


def escanear(img, semente=0, angulo=None, largura=1414, vira=False):
    """Simula o scanner: leve rotação, deslocamento, desfoque, ruído e redução."""
    rnd = random.Random(semente)
    h, w = img.shape[:2]
    ang = rnd.uniform(-1.8, 1.8) if angulo is None else angulo
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[:, 2] += (rnd.uniform(-15, 15), rnd.uniform(-15, 15))
    out = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    out = cv2.GaussianBlur(out, (3, 3), 0)
    ruido = np.random.default_rng(semente).normal(0, 6, out.shape)
    out = np.clip(out.astype(float) + ruido, 0, 255).astype(np.uint8)
    out = cv2.resize(out, (largura, round(h * largura / w)), interpolation=cv2.INTER_AREA)
    if vira:
        out = cv2.rotate(out, cv2.ROTATE_180)
    return out
