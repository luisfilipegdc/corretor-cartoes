"""Leitura óptica do cartão resposta (modelo Gabaritech, com 4 marcadores xadrez nos cantos).

Fluxo: página -> acha os 4 marcadores -> endireita -> acha as bolhas -> monta o layout
(quais bolhas formam cada questão) -> mede o preenchimento de cada bolha.
"""
from __future__ import annotations

import cv2
import numpy as np

W = 1500          # distância, em pixels, entre os centros dos marcadores esquerdo e direito
MX = 60           # margem lateral da imagem endireitada
MT = 330          # margem superior (cabe o cabeçalho com nome, QR etc.)
MB = 50           # margem inferior
LARGURA_PAGINA = 1600

LETRAS = {2: ["C", "E"], 3: ["A", "B", "C"], 4: ["A", "B", "C", "D"], 5: ["A", "B", "C", "D", "E"]}
NOME_TIPO = {2: "CE", 3: "ABC", 4: "ABCD", 5: "ABCDE"}


class ErroLeitura(Exception):
    pass


# ---------------------------------------------------------------- arquivos

def carregar_paginas(caminho: str) -> list[np.ndarray]:
    """Devolve as páginas (BGR) de um PDF ou de uma imagem."""
    if caminho.lower().endswith(".pdf"):
        import pymupdf
        paginas = []
        with pymupdf.open(caminho) as doc:
            for pag in doc:
                pix = pag.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                paginas.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return paginas
    dados = np.fromfile(caminho, dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if img is None:
        raise ErroLeitura("arquivo não é uma imagem nem um PDF legível")
    return [img]


# ---------------------------------------------------------------- alinhamento

def _cinza_normalizado(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if g.shape[1] > g.shape[0]:
        g = cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE)
    esc = LARGURA_PAGINA / g.shape[1]
    interp = cv2.INTER_AREA if esc < 1 else cv2.INTER_CUBIC
    return cv2.resize(g, (LARGURA_PAGINA, round(g.shape[0] * esc)), interpolation=interp)


def _candidatos_marcador(g: np.ndarray) -> list[tuple[float, float, float]]:
    """Quadradinhos com padrão xadrez 2x2 (duas casas escuras na diagonal)."""
    _, bw = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    achados = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (16 <= w <= 50 and 16 <= h <= 50) or not (0.75 <= w / h <= 1.33):
            continue
        q = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                cy, cx = y + h * (0.31 + 0.38 * a), x + w * (0.31 + 0.38 * b)
                y0, y1 = int(round(cy - 0.09 * h)), int(round(cy + 0.09 * h))
                x0, x1 = int(round(cx - 0.09 * w)), int(round(cx + 0.09 * w))
                q[a, b] = bw[y0:y1 + 1, x0:x1 + 1].mean() / 255
        d1, d2 = min(q[0, 0], q[1, 1]), min(q[0, 1], q[1, 0])
        m1, m2 = max(q[0, 0], q[1, 1]), max(q[0, 1], q[1, 0])
        if (d1 > 0.6 and m2 < 0.4) or (d2 > 0.6 and m1 < 0.4):
            achados.append((x + w / 2, y + h / 2, (w + h) / 2))
    return achados


def _modelo_marcador(L: int, invertido: bool) -> np.ndarray:
    """Desenho do marcador com lado L: quadrado escuro com duas casas claras numa diagonal."""
    m = np.zeros((L + 4, L + 4), np.float32)          # 2 px de papel claro em volta
    m[2:-2, 2:-2] = 1
    c = max(3, int(round(0.18 * L)))                   # meia-largura da casa clara
    for fy, fx in ((0.31, 0.31), (0.69, 0.69)):
        if invertido:
            fx = 1 - fx
        cy, cx = 2 + int(round(fy * L)), 2 + int(round(fx * L))
        m[cy - c:cy + c, cx - c:cx + c] = 0
    return m


def _candidatos_xadrez(g: np.ndarray) -> list[tuple[float, float, float]]:
    """Marcadores achados comparando com o desenho deles, em vários tamanhos.

    Mais lento que `_candidatos_marcador`, mas não depende de o marcador estar separado da linha do quadro
    (em JPEG mais comprimido os dois se tocam).
    """
    _, bw = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bw = bw.astype(np.float32)
    achados = []
    faixa = int(bw.shape[1] * 0.27)                     # os marcadores ficam nas laterais da página
    for x0, tira in ((0, bw[:, :faixa]), (bw.shape[1] - faixa, bw[:, -faixa:])):
        for L in range(24, 36, 2):
            for inv in (False, True):
                r = cv2.matchTemplate(tira, _modelo_marcador(L, inv), cv2.TM_CCOEFF_NORMED)
                n, rot, _st, _c = cv2.connectedComponentsWithStats((r >= 0.62).astype(np.uint8), connectivity=8)
                for k in range(1, n):
                    ys, xs = np.where(rot == k)
                    q = int(np.argmax(r[ys, xs]))
                    achados.append((float(r[ys[q], xs[q]]), x0 + xs[q] + (L + 4) / 2, ys[q] + (L + 4) / 2, float(L)))
    achados.sort(reverse=True)                          # melhor encaixe primeiro; descarta repetidos
    unicos: list[tuple[float, float, float]] = []
    for _f, x, y, L in achados:
        if all(abs(x - u[0]) > 20 or abs(y - u[1]) > 20 for u in unicos):
            unicos.append((x, y, L))
    return unicos


def _achar_cantos(g: np.ndarray) -> np.ndarray | None:
    cantos = _quatro_cantos(_candidatos_marcador(g), g.shape)
    if cantos is None:
        cantos = _quatro_cantos(_candidatos_xadrez(g), g.shape)
    return cantos


def _quatro_cantos(cands, forma) -> np.ndarray | None:
    if len(cands) < 4:
        return None
    alt, larg = forma
    pts = np.array([(c[0], c[1]) for c in cands], dtype=np.float32)
    cantos = []
    for cx, cy in ((0, 0), (larg, 0), (larg, alt), (0, alt)):
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        cantos.append(int(np.argmin(d)))
    if len(set(cantos)) < 4:
        return None
    tl, tr, br, bl = (pts[i] for i in cantos)
    topo, base = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    esq, dir_ = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    if topo < larg * 0.5 or esq < alt * 0.3:
        return None
    if abs(topo - base) / topo > 0.08 or abs(esq - dir_) / esq > 0.08:
        return None
    return np.array([tl, tr, br, bl], dtype=np.float32)


def endireitar(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Devolve (imagem cinza endireitada, H) onde H é a altura entre marcadores."""
    g = _cinza_normalizado(img)
    for tentativa in range(2):
        cantos = _achar_cantos(g)
        if cantos is None:
            raise ErroLeitura("não achei os 4 marcadores dos cantos (página cortada, dobrada ou de outro modelo)")
        tl, tr, br, bl = cantos
        larg = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        # o cabeçalho (muito texto) fica acima dos marcadores de cima; se estiver embaixo, a folha está de ponta-cabeça
        faixa = int(larg * 0.17)
        escuro = (g < 128)
        y_top = int(min(tl[1], tr[1]))
        y_bot = int(max(bl[1], br[1]))
        cima = escuro[max(0, y_top - faixa):max(0, y_top - 20)].sum()
        baixo = escuro[min(g.shape[0], y_bot + 20):min(g.shape[0], y_bot + faixa)].sum()
        if tentativa == 0 and baixo > cima * 1.5 and baixo > 2000:
            g = cv2.rotate(g, cv2.ROTATE_180)
            continue
        break
    alt = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    H = int(round(W * alt / larg))
    dst = np.array([(MX, MT), (MX + W, MT), (MX + W, MT + H), (MX, MT + H)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(cantos, dst)
    reta = cv2.warpPerspective(g, M, (W + 2 * MX, H + MT + MB), flags=cv2.INTER_LINEAR, borderValue=255)
    return reta, H


# ---------------------------------------------------------------- bolhas

def _limiar(reta: np.ndarray) -> float:
    miolo = reta[MT + 40:-MB - 40, MX + 30:-MX - 30]
    t, _ = cv2.threshold(miolo, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(np.clip(t, 110, 175))


def _anel(escuro: np.ndarray, cx: float, cy: float, r: float, folga: float = 3.0,
          cinza: np.ndarray | None = None) -> tuple[float, float]:
    """Quanto do contorno de um círculo de raio ~r está impresso (0 a 1) e o raio que melhor encaixa."""
    def pontos(raio):
        xs = np.clip(np.round(cx + raio * _CA).astype(int), 0, escuro.shape[1] - 1)
        ys = np.clip(np.round(cy + raio * _SA).astype(int), 0, escuro.shape[0] - 1)
        return ys, xs

    melhor, melhor_r = 0.0, r
    for rr in np.arange(r - folga, r + folga + 0.5, 1.0):
        f = escuro[pontos(rr)].mean()
        if f > melhor + 0.02 or (f >= melhor - 0.02 and f >= 0.8):   # fica com o raio mais externo que encaixa
            melhor, melhor_r = max(f, melhor), rr
    # uma bolha tem papel claro logo por fora do contorno; a célula cinza do número da questão não tem
    if escuro[pontos(melhor_r + 4.5)].mean() > 0.4:
        return 0.0, melhor_r
    if cinza is not None:
        contraste = float(cinza[pontos(melhor_r + 5)].mean()) - float(cinza[pontos(melhor_r)].mean())
        if contraste < 60:
            return 0.0, melhor_r
    return melhor, melhor_r


_ANG = np.linspace(0, 2 * np.pi, 48, endpoint=False)
_CA, _SA = np.cos(_ANG), np.sin(_ANG)


def achar_bolhas(reta: np.ndarray, H: int) -> np.ndarray:
    """Círculos (x, y, r) na área de respostas, vazios ou preenchidos."""
    thr = _limiar(reta)
    y0, y1 = MT + 45, MT + H - 45
    x0, x1 = MX + 35, MX + W - 35
    area = cv2.medianBlur(reta[y0:y1, x0:x1], 5)
    cs = cv2.HoughCircles(area, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=16,
                          minRadius=12, maxRadius=21)
    if cs is None:
        return np.zeros((0, 3))
    escuro = cv2.dilate((reta < thr).astype(np.uint8), np.ones((3, 3), np.uint8))
    suave = cv2.erode(reta, np.ones((3, 3), np.uint8))   # engrossa o traço para medir o contraste do contorno
    bons = []
    for cx, cy, r in cs[0]:
        f, rr = _anel(escuro, cx + x0, cy + y0, r, cinza=suave)
        if f >= 0.8:
            bons.append((cx + x0, cy + y0, rr))
    if not bons:
        return np.zeros((0, 3))
    bons = np.array(bons)
    R = float(np.median(bons[:, 2]))
    bons = bons[np.abs(bons[:, 2] - R) <= 3.5]
    # completa a grade: procura bolhas que o detector deixou passar ao lado/acima/abaixo das já achadas
    for _ in range(3):
        px, py = _passo_x(bons, R), _passo_y(bons, R)
        novos = []
        for x, y, _r in bons:
            for ex, ey in ((x - px, y), (x + px, y), (x, y - py), (x, y + py)):
                if not (x0 < ex < x1 and y0 < ey < y1):
                    continue
                todos = np.vstack([bons[:, :2]] + ([np.array(novos)[:, :2]] if novos else []))
                if (np.hypot(todos[:, 0] - ex, todos[:, 1] - ey) < 1.5 * R).any():
                    continue
                melhor = (0.0, ex, ey)
                for ox in range(-4, 5, 2):
                    for oy in range(-4, 5, 2):
                        f, _rr = _anel(escuro, ex + ox, ey + oy, R, folga=1.0, cinza=suave)
                        if f > melhor[0]:
                            melhor = (f, ex + ox, ey + oy)
                if melhor[0] >= 0.85:
                    novos.append((melhor[1], melhor[2], R))
        if not novos:
            break
        bons = np.vstack([bons, np.array(novos)])
    return bons


def _passo_y(b: np.ndarray, R: float) -> float:
    ds = []
    for x, y, _ in b:
        m = (np.abs(b[:, 0] - x) < 0.5 * R) & (b[:, 1] - y > 1.5 * R) & (b[:, 1] - y < 4 * R)
        if m.any():
            ds.append((b[m, 1] - y).min())
    return float(np.median(ds)) if ds else 2.75 * R


def _passo_x(b: np.ndarray, R: float) -> float:
    ds = []
    for x, y, _ in b:
        m = (np.abs(b[:, 1] - y) < 0.5 * R) & (b[:, 0] - x > 1.5 * R) & (b[:, 0] - x < 4 * R)
        if m.any():
            ds.append((b[m, 0] - x).min())
    return float(np.median(ds)) if ds else 2.75 * R


def _agrupar(valores: list[float], tol: float) -> list[float]:
    """Mapa valor -> representante do grupo (menor valor do grupo)."""
    rep = {}
    atual = None
    for v in sorted(set(valores)):
        if atual is None or v - ultimo > tol:
            atual = v
        rep[v] = atual
        ultimo = v
    return [rep[v] for v in valores]


def montar_layout(reta: np.ndarray, bolhas: np.ndarray, H: int) -> dict | None:
    """Descobre quais bolhas formam cada questão. Devolve None se não houver bolhas."""
    if len(bolhas) < 2:
        return None
    R = float(np.median(bolhas[:, 2]))
    px = _passo_x(bolhas, R)
    branco = float(np.percentile(reta[MT + 40:MT + H - 40, MX + 30:-MX - 30], 90))

    # fileiras: bolhas vizinhas na horizontal (mesmo y, distância de um passo)
    n = len(bolhas)
    pai = list(range(n))

    def raiz(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i

    ordem = np.argsort(bolhas[:, 0])
    for a in range(n):
        i = ordem[a]
        for bidx in range(a + 1, n):
            j = ordem[bidx]
            dx = bolhas[j, 0] - bolhas[i, 0]
            if dx > 1.4 * px:
                break
            if abs(bolhas[j, 1] - bolhas[i, 1]) < 0.5 * R and dx > 0.6 * px:
                pai[raiz(j)] = raiz(i)
    grupos: dict[int, list[int]] = {}
    for i in range(n):
        grupos.setdefault(raiz(i), []).append(i)

    fileiras = []
    for idx in grupos.values():
        pts = bolhas[sorted(idx, key=lambda k: bolhas[k, 0])][:, :2]
        xa, ya = pts[0, 0], float(pts[:, 1].mean())
        # célula cinza com o número da questão, um passo à esquerda da primeira bolha
        cx0, cx1 = int(xa - px - 0.5 * R), int(xa - px + 0.5 * R)
        cy0, cy1 = int(ya - 0.5 * R), int(ya + 0.5 * R)
        celula = reta[max(cy0, 0):cy1, max(cx0, 0):cx1]
        numerada = celula.size > 0 and float(celula.mean()) < branco - 25
        fileiras.append({"x0": float(xa), "y": ya, "pts": pts, "n": len(pts), "numerada": bool(numerada)})

    unidades = []
    avisos = []
    # questões numeradas: cada fileira é uma questão
    for f in fileiras:
        if f["numerada"]:
            if f["n"] not in LETRAS:
                avisos.append(f"fileira com {f['n']} bolhas em y={f['y']:.0f}")
                continue
            unidades.append({"tipo": NOME_TIPO[f["n"]], "opcoes": LETRAS[f["n"]],
                             "pos": [[float(x), float(y)] for x, y in f["pts"]],
                             "x0": f["x0"], "y0": f["y"], "secao": f["x0"]})
    # blocos numéricos: 10 fileiras (dígitos 0-9) sem célula de número
    soltas = [f for f in fileiras if not f["numerada"]]
    reps = _agrupar([f["x0"] for f in soltas], 0.5 * px)
    blocos = []
    for rep in sorted(set(reps)):
        col = sorted([f for f, r in zip(soltas, reps) if r == rep], key=lambda f: f["y"])
        gaps = np.diff([f["y"] for f in col])
        passo = float(np.median(gaps)) if len(gaps) else 0
        pedacos, atual = [], [col[0]]
        for f, g in zip(col[1:], gaps):
            if g > 1.6 * passo:
                pedacos.append(atual)
                atual = []
            atual.append(f)
        pedacos.append(atual)
        for p in pedacos:
            ns = {f["n"] for f in p}
            if len(p) == 10 and len(ns) == 1:
                nd = p[0]["n"]
                pos = [[[float(f["pts"][c][0]), float(f["pts"][c][1])] for f in p] for c in range(nd)]
                blocos.append({"tipo": "NUM", "digitos": nd, "pos": pos,
                               "x0": p[0]["x0"], "x1": float(p[0]["pts"][-1][0]), "y0": p[0]["y"]})
            else:
                avisos.append(f"bloco sem número com {len(p)} fileiras em x={rep:.0f}")
    # blocos numéricos lado a lado formam uma seção, lida da esquerda para a direita
    blocos.sort(key=lambda b: (round(b["y0"] / (3 * R)), b["x0"]))
    for i, b in enumerate(blocos):
        b["secao"] = b["x0"]
        for ant in blocos[:i]:
            if abs(ant["y0"] - b["y0"]) < 3 * R and 0 < b["x0"] - ant["x1"] < 5 * px:
                b["secao"] = ant["secao"]
    for b in blocos:  # blocos empilhados na mesma coluna ficam na mesma seção
        for o in blocos:
            if abs(o["x0"] - b["x0"]) < 0.5 * px and o["y0"] < b["y0"]:
                b["secao"] = min(b["secao"], o["secao"])
    unidades += blocos
    if not unidades:
        return None
    secs = _agrupar([u["secao"] for u in unidades], 0.6 * px)
    for u, s in zip(unidades, secs):
        u["secao"] = s
    unidades.sort(key=lambda u: (u["secao"], round(u["y0"] / R), u["x0"]))
    questoes = []
    for i, u in enumerate(unidades, 1):
        q = {"n": i, "tipo": u["tipo"], "pos": u["pos"]}
        if u["tipo"] == "NUM":
            q["digitos"] = u["digitos"]
        else:
            q["opcoes"] = u["opcoes"]
        questoes.append(q)
    return {"R": R, "H": H, "questoes": questoes, "avisos": avisos}


def assinatura(layout: dict | None) -> str:
    if not layout:
        return ""
    return "|".join(q["tipo"] + (str(q["digitos"]) if q["tipo"] == "NUM" else "") for q in layout["questoes"])


def _posicoes(layout: dict) -> np.ndarray:
    pts = []
    for q in layout["questoes"]:
        if q["tipo"] == "NUM":
            for col in q["pos"]:
                pts += col
        else:
            pts += q["pos"]
    return np.array(pts, dtype=float)


def aderencia(layout: dict, bolhas: np.ndarray) -> float:
    """Fração das posições do layout que têm uma bolha detectada por perto."""
    if len(bolhas) == 0 or abs(len(_posicoes(layout)) - len(bolhas)) > 0.25 * len(bolhas) + 3:
        return 0.0
    pts = _posicoes(layout)
    d = np.hypot(pts[:, None, 0] - bolhas[None, :, 0], pts[:, None, 1] - bolhas[None, :, 1]).min(axis=1)
    return float((d < 0.6 * layout["R"]).mean())


def deslocamento(layout: dict, bolhas: np.ndarray) -> tuple[float, float]:
    """Pequeno ajuste (dx, dy) do layout de referência para esta folha."""
    if len(bolhas) == 0:
        return 0.0, 0.0
    pts = _posicoes(layout)
    dist = np.hypot(pts[:, None, 0] - bolhas[None, :, 0], pts[:, None, 1] - bolhas[None, :, 1])
    j = dist.argmin(axis=1)
    ok = dist[np.arange(len(pts)), j] < layout["R"]
    if ok.sum() < 3:
        return 0.0, 0.0
    return (float(np.median(bolhas[j[ok], 0] - pts[ok, 0])), float(np.median(bolhas[j[ok], 1] - pts[ok, 1])))


def media_layouts(layouts: list[dict]) -> dict:
    """Layout de referência: mediana das posições de vários cartões com a mesma assinatura."""
    base = layouts[0]
    ref = {"R": float(np.median([l["R"] for l in layouts])), "H": base["H"], "avisos": [], "questoes": []}
    for i, q in enumerate(base["questoes"]):
        pos = np.median(np.array([l["questoes"][i]["pos"] for l in layouts], dtype=float), axis=0)
        nq = dict(q)
        nq["pos"] = pos.tolist()
        ref["questoes"].append(nq)
    return ref


# ---------------------------------------------------------------- leitura das marcações

MARCADA = 0.45
DUVIDA = 0.22


def _preenchimento(reta, thr, x, y, R) -> float:
    r = max(3, int(round(0.62 * R)))
    xi, yi = int(round(x)), int(round(y))
    rec = reta[yi - r:yi + r + 1, xi - r:xi + r + 1]
    if rec.shape != (2 * r + 1, 2 * r + 1):
        return 0.0
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    masc = xx * xx + yy * yy <= r * r
    return float((rec[masc] < thr).mean())


def ler(reta: np.ndarray, layout: dict, dx: float = 0.0, dy: float = 0.0) -> list[dict]:
    """Resposta lida em cada questão.

    resposta: 'C', 'E', 'A'..'E', '011' (numérica); '' em branco; 'A+C' para dupla marcação;
    '_' marca dígito em branco dentro de uma numérica.
    """
    thr = _limiar(reta)
    R = layout["R"]
    saida = []
    for q in layout["questoes"]:
        if q["tipo"] == "NUM":
            txt, dupla, duvida, fills = "", False, False, []
            for col in q["pos"]:
                f = [_preenchimento(reta, thr, x + dx, y + dy, R) for x, y in col]
                fills.append(f)
                marc = [d for d, v in enumerate(f) if v >= MARCADA]
                if any(DUVIDA <= v < MARCADA for v in f):
                    duvida = True
                if len(marc) == 1:
                    txt += str(marc[0])
                elif not marc:
                    txt += "_"
                else:
                    txt += "?"
                    dupla = True
            branco = set(txt) == {"_"}
            saida.append({"n": q["n"], "tipo": "NUM", "resposta": "" if branco else txt,
                          "dupla": dupla, "duvida": duvida or (not branco and "_" in txt), "fills": fills})
        else:
            f = [_preenchimento(reta, thr, x + dx, y + dy, R) for x, y in q["pos"]]
            marc = [q["opcoes"][i] for i, v in enumerate(f) if v >= MARCADA]
            duvida = any(DUVIDA <= v < MARCADA for v in f)
            saida.append({"n": q["n"], "tipo": q["tipo"], "resposta": "+".join(marc),
                          "dupla": len(marc) > 1, "duvida": duvida, "fills": f})
    return saida


# ---------------------------------------------------------------- imagem de conferência

VERDE, VERMELHO, AZUL, CINZA = (60, 160, 40), (40, 40, 220), (220, 130, 0), (150, 150, 150)


def conferencia(reta: np.ndarray, layout: dict, respostas: list[dict], correcao: list[dict] | None,
                dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Cartão endireitado com as marcações lidas contornadas (verde certo, vermelho errado, azul = gabarito)."""
    img = cv2.cvtColor(reta, cv2.COLOR_GRAY2BGR)
    R = int(round(layout["R"])) + 4
    por_n = {c["n"]: c for c in (correcao or [])}
    for q, r in zip(layout["questoes"], respostas):
        c = por_n.get(q["n"])
        st = c["status"] if c else None
        cor = {"certa": VERDE, "errada": VERMELHO, "dupla": VERMELHO}.get(st, CINZA if c else VERDE)
        gab = (c or {}).get("gabarito", "")
        if q["tipo"] == "NUM":
            lido = r["resposta"].ljust(q["digitos"], "_") if r["resposta"] else "_" * q["digitos"]
            for ci, col in enumerate(q["pos"]):
                for d, (x, y) in enumerate(col):
                    p = (int(round(x + dx)), int(round(y + dy)))
                    marcado = (r["fills"][ci][d] >= MARCADA) if "fills" in r else lido[ci] == str(d)
                    if marcado:
                        cv2.circle(img, p, R, cor, 3)
                    elif st in ("errada", "branco", "dupla") and len(gab) == q["digitos"] and gab[ci] == str(d):
                        cv2.circle(img, p, R, AZUL, 2)
            x, y = q["pos"][0][0]
            cv2.putText(img, str(q["n"]), (int(x + dx - R), int(y + dy - 2.2 * R)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, AZUL, 2, cv2.LINE_AA)
        else:
            marcadas = set(r["resposta"].split("+")) if r["resposta"] else set()
            for letra, (x, y) in zip(q["opcoes"], q["pos"]):
                p = (int(round(x + dx)), int(round(y + dy)))
                if letra in marcadas:
                    cv2.circle(img, p, R, cor, 3)
                elif st in ("errada", "branco", "dupla") and letra == gab:
                    cv2.circle(img, p, R, AZUL, 2)
            x, y = q["pos"][-1]
            cv2.putText(img, str(q["n"]), (int(x + dx + 1.6 * R), int(y + dy + 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, AZUL, 2, cv2.LINE_AA)
    return img
