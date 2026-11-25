"""Leitura de uma página em processo separado, para ler lotes grandes usando vários núcleos."""
from __future__ import annotations

import uuid

import cv2

from . import cabecalho, omr


def contar_paginas(caminho: str) -> int:
    if caminho.lower().endswith(".pdf"):
        import pymupdf
        with pymupdf.open(caminho) as doc:
            return len(doc)
    return 1


def _pagina(caminho: str, indice: int):
    if caminho.lower().endswith(".pdf"):
        import numpy as np
        import pymupdf
        with pymupdf.open(caminho) as doc:
            pix = doc[indice].get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return omr.carregar_paginas(caminho)[0]


def ler_imagem(img, indice: int, nome: str, pasta_retas: str) -> dict:
    """Endireita, guarda a imagem e acha as bolhas de uma página já aberta. Não lança exceção.

    O cabeçalho (nome, turma...) é lido depois.
    """
    c = {"id": uuid.uuid4().hex[:12], "arquivo": nome, "pagina": indice + 1, "info": {}, "respostas": [],
         "avisos": [], "erro": "", "editado": False, "layout": ""}
    item = {"c": c, "bolhas": None, "lay": None, "reta": None}
    try:
        if callable(img):
            img = img()
        if img is None:
            raise omr.ErroLeitura("arquivo não é uma imagem nem um PDF legível")
        reta, H = omr.endireitar(img)
        cv2.imwrite(f"{pasta_retas}/{c['id']}.jpg", reta, [cv2.IMWRITE_JPEG_QUALITY, 88])
        item["bolhas"] = omr.achar_bolhas(reta, H)
        item["lay"] = omr.montar_layout(reta, item["bolhas"], H)
        if item["lay"] is None:
            raise omr.ErroLeitura("não achei as bolhas de resposta")
    except omr.ErroLeitura as ex:
        c["erro"] = str(ex)
    except Exception as ex:  # arquivo ilegível vira um cartão com erro, para aparecer na lista
        c["erro"] = f"falha ao ler: {ex}"
    return item


def ler_pagina(caminho: str, indice: int, nome: str, pasta_retas: str) -> dict:
    return ler_imagem(lambda: _pagina(caminho, indice), indice, nome, pasta_retas)


def _cabecalho_de_arquivo(caminho: str) -> dict:
    reta = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    return cabecalho.ler(reta) if reta is not None else {}


class LeitorCabecalhos:
    """Lê o cabeçalho (OCR) de imagens endireitadas em poucos processos próprios, com tempo limite.

    O OCR do macOS às vezes trava quando muitos processos o usam ao mesmo tempo. Se um travar, os processos
    são encerrados e a página é tentada mais uma vez; se travar de novo, fica sem identificação (o cartão
    aparece para conferência) e o lote continua.
    """

    def __init__(self, processos: int = 2, limite: float = 25.0):
        self.processos, self.limite, self.pool = processos, limite, None

    def ler(self, caminhos: list[str], ao_terminar=None) -> dict[str, dict]:
        resultado: dict[str, dict] = {}
        tentativas: dict[str, int] = {}
        pendentes = list(caminhos)
        while pendentes:
            from concurrent.futures import ProcessPoolExecutor
            from concurrent.futures import TimeoutError as Demorou
            from concurrent.futures.process import BrokenProcessPool
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.processos)
            futuros = [(c, self.pool.submit(_cabecalho_de_arquivo, c)) for c in pendentes]
            travou = False
            for c, f in futuros:
                try:
                    if travou and not f.done():
                        continue
                    resultado[c] = f.result(timeout=self.limite)
                except (Demorou, BrokenProcessPool):
                    travou = True
                    tentativas[c] = tentativas.get(c, 0) + 1
                    if tentativas[c] >= 2:
                        resultado[c] = {}
                except Exception:
                    resultado[c] = {}
                if c in resultado and ao_terminar:
                    ao_terminar()
            if travou:
                self.fechar(matar=True)
            pendentes = [c for c in pendentes if c not in resultado]
        return resultado

    def fechar(self, matar: bool = False):
        if self.pool is None:
            return
        if matar:
            for proc in list(getattr(self.pool, "_processes", {}).values()):
                proc.kill()
        self.pool.shutdown(wait=not matar, cancel_futures=True)
        self.pool = None


def ler_cabecalhos(caminhos: list[str], ao_terminar=None) -> dict[str, dict]:
    leitor = LeitorCabecalhos()
    try:
        return leitor.ler(caminhos, ao_terminar)
    finally:
        leitor.fechar()
