import re

# ~ resultado = re.search("c", "abcdefc")

# ~ resultado = re.findall("e", "esta es una cadena.")

# ~ resultado = re.split("\s", "esta es una cadena.")

# ~ resultado = re.sub("\s", "\n", "esta es una cadena.")

# ~ patron = re.compile(",")
# ~ resultado = patron.findall("Cadena1, Cadena2, Cadena3, Cadena4, Cadena5")
# ~ print(resultado)
# ~ resultado2 = patron.split("Cadena1, Cadena2, Cadena3, Cadena4, Cadena5")
# ~ print(resultado2)

patron = re.compile("\d+\.?\d*")
resultado = patron.findall("Esta es una cadena con los números 14, 15.5 y 0.25, 8.")

print (resultado)
