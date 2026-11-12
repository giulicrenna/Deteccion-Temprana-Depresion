from pypdf import PdfReader

# 1. Ruta de tu archivo PDF
ruta_pdf = '../informe/main.pdf'
ruta_txt = 'salida.txt'

# 2. Leer el PDF
lector = PdfReader(ruta_pdf)
texto_completo = []

# 3. Recorrer cada página y extraer el texto
for i, pagina in enumerate(lector.pages):
    texto_extraido = pagina.extract_text()
    if texto_extraido:
        texto_completo.append(texto_extraido)

# 4. Guardar en un archivo de texto plano
with open(ruta_txt, 'w', encoding='utf-8') as archivo_salida:
    archivo_salida.write('\n'.join(texto_completo))

print(f'¡Texto extraído con éxito en "{ruta_txt}"!')
