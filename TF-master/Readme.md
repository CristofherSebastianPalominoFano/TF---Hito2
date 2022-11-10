Objetivos
• Desarrollar la competencia general de razonamiento cuantitativo y la competencia específica de uso de técnicas y herramientas acorde a los objetivos del curso.

• Desarrollar un algoritmo que permita resolver completa o parcialmente el problema del vendedor viajero.

• Determinar la importancia de la aplicación de algoritmos eficientes a la hora de resolver un problema.

• Analizar la eficiencia y complejidad de los algoritmos planteados.

Marco Teórico
Algoritmo de Prim
El resultado ejecutar el algoritmo de Prim es un arbol de expansion minima, dado un grafo no dirigido y conexo, es decir encuentra un subconjunto de las aristas que forman un arbol que incluye los vertices del grafo inicial, donde el peso total de las aristas es la mínima posible.

Funcionamiento
Se marca un vértice cualquiera. Será el vértice de partida.
Se selecciona la arista de menor peso incidente en el vértice seleccionado anteriormente y se selecciona el otro vértice en el que incide dicha arista.
Repetir el paso 2 siempre que la arista elija enlace un vértice seleccionado y otro que no lo esté. Es decir, siempre que la arista elegida no cree ningún ciclo.
El árbol de expansión mínima será encontrado cuando hayan sido seleccionados todos los vértices del grafo.
Complejidad
El bucle principal se. ejecuta n- 1 veces, en cada iteración cada bucle interior toma O(n), por lo tanto el tiempo de ejecución del algoritmo de PRIM toma O(n^2 ) .

Algoritmo Kruskal
Primero que nada, debemos definir los algoritmos de unión y encontrar:

Union Find es una estructura de datos que modela una colección de conjuntos disjuntos (disjoint-sets) y está basada en 2:

Find( A ): Determina a cual conjunto pertenece al elemento A. Esta operación puede ser usada para determinar si 2 elementos están o no en el mismo conjunto.
Union( A, B ): Une todo el conjunto al que pertenece A con todo el conjunto al que pertenece B, dando como resultado un nuevo conjunto basado en los elementos tanto de A como de B.
Método Buscar:
Como se explica al inicio de este método determina a qué componente conexa pertenece a un vértice X determinado, ello lo hace retornando el vértice raíz del árbol en el que se encuentra el vértice X.

Método Unión:
Como se explica al inicio este método me permite unir 2 componentes conectados, ello se realiza por lo siguiente:

Obtenemos la raíz del vértice x.
Obtenemos la raíz del vértice y.
Actualizamos el padre de alguna de las raíces, asignándole como nuevo padre la otra raíz.
Funcionamiento Kruskal
Primeramente ordenaremos las aristas del grafo por su peso de menor a mayor. Mediante la técnica greedy Kruskal intentara unir cada arista siempre y cuando no se forme un ciclo, ello se realizará mediante los métodos Union-Find. Como hemos ordenado las aristas por peso comenzaremos con la arista de menor peso, si los vértices que contienen dicha arista no están en la misma componente conexa entonces los unimos para formar una sola componentes mediante Unión.

Complejidad
La complejidad se puede analizar contemplando lo siguiente:

• Se tiene una complejidad de O(a log(a)) para ordenar los arcos, el cual es equivalente a O(a log(a)), como n- 1 <= a <= n(n- 1)/2

• Se tiene una complejidad de O(n) al inicializar los n conjuntos disjuntos

• Se tiene una complejidad de 0((2a- 1)log*n para todas las búsquedas

considerando que:

K llamadas a operaciones buscan el líder del conjunto que contiene un vértice de conjuntos disjuntos den elementos lleva un tiempo de O(Klog*n).

log n EO(logn), pero logn ~ O(log n).

Por lo tanto podemos concluir que la complejidad del algoritmo de Kruskal es O(a*log(n)).

Algoritmo Floyd-Warshall
Fue descrito por primera vez por Bernard Roy en 1959, se trata de un algoritmo de análisis sobre gráficos para encontrar el camino mínimo en gráficos dirigidos. Este algoritmo encuentra el mejor camino de todos los pares de vértices en una sola ejecución y es un claro ejemplo de programación dinámica.

Este algoritmo a su vez se trata de la mezcla de dos algoritmos distintos, los cuales son indicados en su nombre compuesto; así pues, para entender aún más el funcionamiento de este algoritmo se sigue una breve explicación de los algoritmos que lo componen:

Algoritmo de Warshall
Es un algoritmo booleano, ya que hace uso de una matriz compuesta de 0 e 1, los cuales indican que hay o no correspondencia en el grafo y, a travez de este algoritmo se obtiene una matriz transitiva la cual muestra si es que dos nodos se hallarán conexión mediante una unión indirecta.

Algoritmo de Floyd
Es bastante similar al algoritmo usado por Warshall, pero este permite el uso de grafos ponderados, permitiendo que la "flecha" que indica la union entre dos nodos tenga valores enteros o infinito, siendo infinito un indicador de que esos nodos no poseen una union directa entre ellos. De esta forma, este algoritmo es perfectamente aplicable para las distancias ocultas entre cada par de nodos que se encuentran conectados.

Funcionamiento Floyd-Warshall
Este algoritmo compara todos los posibles caminos entre cada par de vértices que se encuentra en el grafo en tan solo V^3 comparaciones, lo cual es logrado gracias a que poco a poco hace una estimación del mejor camino entre dos vértices, hasta que se sabe la estimacion optima.

Se define un grafo G con vertices V numerados de 1 a N, y una funcion CaminoMinimo(i, j, k) que devuelve el camino minimo de iaj (los cuales conforman V) utilizando solo los vertices de 1 ak como puntos intermedios en el camino. Dada esta función se procede a calcular el camino mínimo de iaj utilizando solo los vértices de 1 hasta a k+1.

Una vez definido esto, se pueden presentar dos situaciones posibles; el camino mínimo se puede hallar directamente mediante la función CaminoMinimo(i, j, k) y se halla comprendido entre los vértices 1 a k+1; o se encuentra como el camino minimo de k+1 aj, por lo cual se debe concatenar dos caminos minimos para formar el mas optimo.

Complejidad de Floyd-Warshall
La complejidad de este algoritmo es O(n^3) . El algoritmo resuelve eficientemente la búsqueda de todos los caminos más cortos entre posibles nodos. Sin embargo, la busqueda se vuelve lenta.

experimentacion
Solución 1: Aplicando el algoritmo Kruskal
Para esta solución utilizaremos el algoritmo de kruskal para hallar árboles de expansión mínima. En primer lugar, se obtendrán los datos de todos los centros poblados desde un archivo con extensión csv. Para ello definimos el siguiente modelo:

	#Clase CentroPoblado
	class  CentroPoblado:
		def  __init__(self,codigo,nombre,departamento,provincia,distrito,capital,coordX, coordY):
			self.codigo = codigo
			self.nombre = nombre
			self.departamento = departamento
			self.provincia = provincia
			self.distrito = distrito
			self.capital = capital
			self.coordX = coordX
			self.coordY = coordY
		def  __str__(self):
			return  "%s D: %s P: %s D: %s Cap: %d Cod: %s X: %f Y: %f"  % (self.nombre, self.departamento, self.provincia, self.distrito, self.capital, self.codigo,self.coordX,self.coordY)
Luego, definimos una función para leer el conjunto de datos:

	def  leerDataSet(nombreArchivo,inicio,fin  =  0):
		centrosPoblados = []
		try:
			archivo =  open(nombreArchivo)
			lineas = archivo.readlines()
			if fin !=  0:
				lineas = lineas[inicio:(fin+1)]
			else:
				lineas = lineas[inicio:]
			registros = []
			for linea in lineas:
				aux = linea.replace('\n','')
				if  len(aux.split(',')) ==  18: #valida si el registro tiene 18 columnas
					registros.append(aux)
			print(len(registros))
			for registro in registros:
				datos = registro.split(',')
				departamento,provincia,distrito,codigo,nombre = datos[1:6]
				capital =  int(datos[7])
				coordX,coordY = datos[15:17]
				coordX =  float(coordX)
				coordY =  float(coordY)
					centrosPoblados.append(CentroPoblado(codigo,nombre,departamento,provincia,distrito,capital,coordX,coordY))
		except  FileNotFoundError:
			print("Archivo no encontrado.")
		finally:
			archivo.close()
			return centrosPoblados
Implementamos el algoritmo de Kruskal en python:

def  kruskal(G,id):
	aristas = []
	resultado = []
	for arista in G:
		costo,nodo,vecino = arista
		hq.heappush(aristas,(costo,nodo,vecino))
	while  len(aristas):
		costo,u,v = hq.heappop(aristas)
		pu = find(id,u)
		pv = find(id,v)
		if pu != pv:
			resultado.append((costo,u,v))
			union(id,pu,pv)
	return resultado
Finalmente, aplicamos el algoritmo para una muestra:

import matplotlib.pyplot as plt
from Leer import leerDataSet,leerArchivo
from generarGrafo import generarGrafo
centrosPoblados = leerDataSet("dataset.csv",1)
tipoMuestra = {
'RESTANTES':0,
'DEPARTAMENTALES':1,
'PROVINCIALES':2,
'DISTRITALES':3
}
muestra = [] #lista centros poblados
id  = {}
for cep in centrosPoblados:
	if cep.capital == tipoMuestra['DEPARTAMENTALES']:
		id[cep.codigo] = cep.codigo
		muestra.append(cep)
G = generarGrafo(muestra)
arbolExpMin = kruskal(G,id)
print(arbolExpMin)
Resultados
Estos fueron los resultados obtenidos durante la experimentación:

Aplicar su solución a las 25 capitales departamentales. Pruebas departamentales
Aplicar su solución a las 171 capitales provinciales pruebas provinciales
Aplicar su solución a las 1'678 capitales distritales. Prueba en distribuciones
Aplicar su solución a los 143'351 centros poblados restantes No fue posible aplicar este algoritmo en un tiempo razonable para dicha muestra.
Solución 2: Aplicando el algoritmo de Prim
Para la solucion 2 se utilizo el algoritmo Prim:

En primer lugar se crearon diccionarios para las distancias, padres y visitados. Luego se creo un arreglo vacío y en él se agregaron las distancias y los vértices.

def prim(G,centrosPoblados,inicio = 0):
    #lista de aristas
    resultado = []

    dist = {}
    for cep in centrosPoblados:
        dist[cep.codigo] = math.inf
    padres = {}
    for cep in centrosPoblados:
        padres[cep.codigo] = ''
    visitados = {}
    for cep in centrosPoblados:
        visitados[cep.codigo] = False
    q = []
    hq.heappush(q, (0,centrosPoblados[inicio].codigo))
Luego se implemento el algoritmo Prim, donde se retorna un grafo de padres, distancias y el grafo resultante.

	while len(q) > 0:
        _,u = hq.heappop(q)
        if not visitados[u]:
            visitados[u] = True
            for w,v in G[u]:
                if not visitados[v] and w < dist[v]:
                    dist[v] = w
                    padres[v] = u
                    resultado.append((u,v))
                    hq.heappush(q, (w,v))
    return padres,dist,resultado
Finalmente se uso el algoritmo para una muestra:

    centrosPoblados = leerDataSet("dataset.csv",1)
    tipoMuestra = {
        'RESTANTES':0,
        'DEPARTAMENTALES':1,
        'PROVINCIALES':2,
        'DISTRITALES':3
    }
    muestra = []
    for cep in centrosPoblados:
        if cep.capital == tipoMuestra['DEPARTAMENTALES']:
            muestra.append(cep)
    grafo = generarGrafo(muestra)
    padres,distancias,resultado = prim(grafo,muestra)
    print(resultado)
Resultados
Estos fueron los resultados obtenidos durante la experimentación:

Aplicar su solución a las 25 capitales departamentales. Pruebas departamentales
Aplicar su solución a las 171 capitales provinciales pruebas provinciales
Aplicar su solución a las 1'678 capitales distritales. Prueba en distribuciones
Aplicar su solución a los 143'351 centros poblados restantes No fue posible aplicar este algoritmo en un tiempo razonable para dicha muestra.
Solución 3: Aplicando el algoritmo Floyd-Warshall
Para esta solución se implementará una matriz que almacene todos los mejores caminos para llegar a todos los nodos con que se cuenta. En primer lugar, se generará un grafo común utilizando el conjunto de datos creado anteriormente:

def generarGrafo(centrosPoblados):
   grafo = []
   
   for cep in centrosPoblados:
       copia = centrosPoblados[:] # genera una copia de todos los centros poblados
       copia.remove(cep) #removemos el cep seleccionado para no generar un camino al mismo punto
       destinosPorCep = random.randint(3,5) # seleccionamos entre 3 y 5 destinos por cada centro poblado
       for _ in range(destinosPorCep):
           destino = random.choice(copia) #seleccionamos un destino al azar
           copia.remove(destino) #removemos el destino para que no se repita
           #Generamos una nueva arista
           distancia = calcularDistancia(cep,destino)
           arista = (distancia,cep.codigo,destino.codigo)
           grafo.append(arista)
   return grafo
Ya que se va a trabajar con matrices, este grafo generado se ve cambiado para fines de mejor manejo, reemplazando los codigos reales de los nodos por numeros mas pequenos desde 0 a n-1 nodos:

def generarGrafoFloyd(muestra):
    G=generarGrafo(muestra)
    n=len(G)
    grafo=[[] for _ in range(n)]
    
    i=0
    r=[]
    for _, nodo, _ in G:
        r.append(int(nodo))
        
    norep=repetidos(r)


    uCodigos=[]
    for nodo in norep:
        uCodigos.append((i, nodo))
        i=i+1

    for u in range(n):
        distancias=[x[0] for x in G]
        nodos=[x[1] for x in G]
        vecinos=[x[2] for x in G]
        nd=buscar(uCodigos, int(nodos[u]))
        v=buscar(uCodigos, int(vecinos[u]))
        if distancias[u]==None:
            distancias[u]=0
        grafo[u].append((distancias[u], nd, v))
        
    return grafo, uCodigos
De esta función se obtiene un nuevo grafo que ya puede ser procesado por nuestro algoritmo Floyd-Warshall y un arreglo que contiene los códigos reales y sus números ahora asignados. A continuación se utiliza el algoritmo de Floyd-Warshall para obtener la matriz con todos los caminos posibles:

def floydWarshall(G, tamano):
    n=len(G)
    maCostos = [[math.inf]*tamano for _ in range(tamano)]
    maPadres = [[-1]*tamano for _ in range(tamano)]
    for nodos in range(n):
        for distancia, nodo, vecino in G[nodos]:
            maCostos[nodo][nodo] = 0
            maCostos[nodo][vecino] = distancia
            maPadres[nodo][vecino] = nodo
            
    for k in range(tamano):
        for i in range(tamano):
            for j in range(tamano):
                pesoAcumulado = maCostos[i][k] + maCostos[k][j]
                if maCostos[i][j] > pesoAcumulado:
                    maCostos[i][j] = pesoAcumulado
                    maPadres[i][j] = k
           
    return maPadres
Finalmente se aplica el algoritmo, obteniendo así la matriz deseada con los resultados:

if __name__ == "__main__":
    centrosPoblados = leerDataSet("dataset.csv",1)
    tipoMuestra = {
        'RESTANTES':0,
        'DEPARTAMENTALES':1,
        'PROVINCIALES':2,
        'DISTRITALES':3
    }
    muestra = [] #lista centros poblados
    id = {}
    for cep in centrosPoblados:
        if cep.capital == tipoMuestra['DEPARTAMENTALES']: #es capital departamental
            id[cep.codigo] = cep.codigo
            muestra.append(cep)
            
    G, uCodigos = generarGrafoFloyd(muestra)
    tamano=len(muestra)
    caminoFloyd = floydWarshall(G, tamano)
    print(caminoFloyd)
Conclusiones
Finalmente, se puede concluir que al momento de la ejecución de los algoritmos Prim y Kruskal:

• Ambos algoritmos se pueden ejecutar de manera satisfactoria y rápida para las muestras de Departamentos, Provincias y Distrital

• Si se ejecuta el algoritmo para las muestras restantes, es decir para todos los centros poblados, se mostraria los resultados adecuados. Sin embargo, el tiempo de ejecucion seria muy extenso, lo cual no es lo mas optimo.
