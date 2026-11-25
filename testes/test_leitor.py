"""Testa a leitura em cartões sintéticos: `.venv/bin/python -m testes.test_leitor [pasta_para_salvar_imagens]`"""
import sys

import cv2

from leitor import omr
from testes.gerar_sintetico import escanear, gerar


def rodar(pasta=None):
    total_erros = 0
    for modelo in ("mat", "hum", "hum2"):
        erros = tot = 0
        for s in range(8):
            img, verdade = gerar(modelo, semente=s)
            sc = escanear(img, semente=s, vira=(s == 3), largura=1414 if s % 2 else 1240)
            reta, H = omr.endireitar(sc)
            b = omr.achar_bolhas(reta, H)
            lay = omr.montar_layout(reta, b, H)
            r = omr.ler(reta, lay)
            lido = {x["n"]: x["resposta"] for x in r}
            dif = {n: (verdade[n], lido.get(n)) for n in verdade if verdade[n] != lido.get(n)}
            erros += len(dif)
            tot += len(verdade)
            print(modelo, s, "bolhas", len(b), "questões", len(lay["questoes"]), "avisos", lay["avisos"], "dif", dif)
            if pasta and s == 0:
                cv2.imwrite(f"{pasta}/sint_{modelo}.png", sc)
                cv2.imwrite(f"{pasta}/conf_{modelo}.jpg", omr.conferencia(reta, lay, r, None))
        print(modelo, "erros", erros, "de", tot)
        total_erros += erros
    return total_erros


if __name__ == "__main__":
    sys.exit(1 if rodar(sys.argv[1] if len(sys.argv) > 1 else None) else 0)
