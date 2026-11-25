// Roda o corretor (o mesmo código Python) dentro do navegador, num processo à parte da página.
// Usado quando não há servidor, por exemplo no GitHub Pages. Nenhum cartão sai do computador.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const FONTES = ["app.py", "leitor/__init__.py", "leitor/omr.py", "leitor/gabarito.py", "leitor/cabecalho.py", "leitor/lote.py"];
let py = null, chamarPy = null;
const aviso = texto => postMessage({tipo: "status", texto});
const sincronizar = carregar => new Promise((ok, falha) => py.FS.syncfs(carregar, e => e ? falha(e) : ok()));

async function iniciar(base) {
  aviso("Baixando o motor do corretor…");
  py = await loadPyodide();
  aviso("Baixando as bibliotecas de imagem…");
  await py.loadPackage(["numpy", "opencv-python", "micropip"]);
  aviso("Baixando as bibliotecas de planilha…");
  await py.runPythonAsync(`import micropip\nawait micropip.install(["flask", "openpyxl"])`);
  aviso("Carregando o corretor…");
  for (const d of ["/app", "/app/leitor", "/dados"]) py.FS.mkdirTree(d);
  py.FS.mount(py.FS.filesystems.IDBFS, {}, "/dados");   // cartões e notas ficam guardados neste navegador
  await sincronizar(true);
  for (const f of FONTES) {
    const r = await fetch(new URL(f, base), {cache: "no-cache"});
    if (!r.ok) throw new Error("não consegui baixar " + f);
    py.FS.writeFile("/app/" + f, await r.text());
  }
  py.runPython(`
import os, sys
os.environ["DADOS"] = "/dados"
sys.path.insert(0, "/app")
os.chdir("/app")
import app
_cliente = app.app.test_client()

def chamar(metodo, rota, corpo, tipo):
    if corpo is not None and not isinstance(corpo, (str, bytes)):
        corpo = bytes(corpo.to_py())
    r = _cliente.open(rota, method=metodo, data=corpo, content_type=tipo)
    return r.status_code, r.mimetype, r.get_data()
`);
  chamarPy = py.globals.get("chamar");
}

onmessage = async ev => {
  const m = ev.data;
  try {
    if (m.tipo === "iniciar") {
      await iniciar(m.base);
      postMessage({id: m.id, ok: true});
      return;
    }
    const res = chamarPy(m.metodo, m.rota, m.corpo ?? null, m.tipoCorpo ?? null);
    const [status, mime, dados] = res.toJs();
    res.destroy();
    if (m.metodo !== "GET") await sincronizar(false);
    postMessage({id: m.id, ok: true, status, mime, dados}, [dados.buffer]);
  } catch (e) {
    postMessage({id: m.id, ok: false, erro: String(e && e.message || e)});
  }
};
