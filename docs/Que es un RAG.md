# RAG

##  ¿Que es un sistema RAG?

RAG son eas siglas de **"Retrieval Augmented Generation"** que significa 
"Generacion Aumentada por Recuperacion". Este es un enfoque que combina la
generacion de texto mediante modelos de lenguaje con información relevante
que puede ser extraída de una **Base de datos** o un **Corpus de Documentos**.
Su objetivo se encuentra en mejorar la calidad y el contenido de las respuestas
de estos modelos de lenguaje, proporcionando información específica y
actualizada, que el modelo por si solo puede no tener.

## ¿Cómo funciona un sistema RAG?

Repasemos la estructura más basica de un sistema RAG, cuál es su flujo y como
debería este funcionar. A continuacion se presenta un gráfico que muestra la 
arquitectura tradicional de un sistema RAG.

![Esta imagen presenta la estructura, 
componentes y flujo de un sistema 
RAG basico.](./img/Estructura-de-un-sistema-RAG.png "Estructura de un RAG basico")

Expliquemos mejor la imagen para entender el funcionamiento de estos sistemas y
sus partes:

1.  **Documentos:** Este es el contexto que se desea aumentar al LLM, es decir, 
esta es la información personal, específica, actualizada, o similar,
que por defecto nuestro LLm no conoce. Puede ser de tipo PDF, texto, otros
similares, y en algunos casos de agentes multimodales, pueden ser imagenes.
2. **Chunks:** Conocidos como **"Chunks", "Partes", "Nodos", "Fragmentos"** es una 
pequeña porcion de un documento largo, que ha sido extraída.
3. **Embeddings:** Los embeddings o incrustaciones buscan convertir el texto 
natural a formato numerico. Para esto, toman un chunk y lo convierten en un
vector de incrustación. Esto es especialmente util, al momento de buscar la
similitud de los documentos, ya que esto nos permite trabajar con las diversas
fórmulas matematicas de distancia. Cuando se realiza una busqueda de similitud
se envia al LLM la pregunta del usuario, en conjunto con las recuperaciones
de similitud, para enriquecer el contexto.

## ¿Por que Agentes RAG, en vez de un RAG basico?

El modelo anteriormente presentado es muy bueno para problemas de "Pregunta-Respuesta" 
**(QA - Query-Answer)** simples, donde se tiene uno o unos pocos documentos. Sin
embargo, puede no llegar a ser suficiente para tareas complejas de QA o de resumen
de un extenso grupo de documentos.

Es en estos escenarios complejos donde los **"Agentes RAG"** entran en escena, 
permitiendo que estas tareas complejas sean manejadas de una manera mucho más simple.
El modelo de agentes permite la incorporacion de llamadas a herramientas (tools) 
dentro del sistema RAG, donde estás tools, pueden ser cualquier tipo de funciones
definidas por nosotros mismo o por alguien más para ser utilizadas y **reutilizadas**.

### A travez de esta documentacion se explicara:

1. **Motores de enrutamiento de consultas:** Este es el modelo más elemental de un agente
RAG. Este modelo brinda la oportunidad de que, a travez de **declaraciones logicas** se
pueda guiar o instruir al modelo LLM sobre qué camino tomar, que acciones realizar y qué 
herramientas utilizar para resolver una tarea específica.
2. **LLamada a herramientas:** Para cuál sea la arquitectura de Agente RAG que se desarrolle
la llamada a herramientas personalizadas, es una forma de incrustar o añadir al sistema
habilidades unicas. Esto se logra al proporcionar al LLM una interfaz o un medio por el cual
el agente pueda seleccionar y utilizar estas herramientas, del el conjunto proporcionado; y 
permitiéndole a este, entregar a la herramienta los parametros que le sean necesarias
para poder ejecutarse.
3. Y por último como hacer que estos **Agentes RAG tengan capacidades de razonamiento de 
varios pasos** asi como el permitirles trabajar con varios documentos.