# Homework

# Part 1: Adversarial attacks (2 unidades)

En esta sección se debe realizar un adversarial attack usando el algoritmo PGD. La idea de esta sección es evidenciar cómo afecta el ataque a un modelo específico, 
así como la influencia que tienen distintas variables como el epsilon, el step size y la norma en el rendimiento del ataque. Para realizar esta parte de la actividad, primero descargue la base de datos de Calthech reorganizada y un modelo preentrenado en éste link y agregue estos dos archivos junto con los script. Luego ejecute el archivo AAAAAAAAAAAAAAAAA y observe el accuracy del modelo antes del ataque y después del ataque (baseline). Habiendo realizado este proceso, realice 3 experimentos adicionales modificando las variables ya mencionadas (epsilon, step size y la norma). Es importante mencionar que estas variables se encuentran declaradas en las primeras líneas del archivo AAAAAAAAAAAAAAA.py.


# Part 2: Adversarial training (2 unidades)

En esta sección se evaluará el efecto de el entrenamiento adversario como defensa en contra de ataques adversarios. Para ello, modifique los parámetros del archivo AAAAAAAAAAAAAAaa.py (primera parte), reemplazandolos con los mejores parámetros encontrados en el primer punto y ejecute el archivo AAAAAAAAAAAAAAAAA.py. Compare el valor obtenido por el modelo después del entrenamiento adversario, con el obtenido previo al mismo.


# Part 3: APGD (1 unidad)

La idea de esta sección es entender el código y pseudocódigo del algoritmo APGD. Para esto, complete el #TO DO en el archivo AAAAAAAAAAAAAAAAAAAAAAAAA. Es recomendable que no utilice ciclos, sino aproximadamente dos lineas de código con lógica tensorial. En el archivo se detalla en español el proceso que se tiene que programar. Habiendo completado el #TO DO, ejecute el archivo y reporte la gráfica generada en el reporte. 

# Bono (0.5 unidades): 

Responda en su reporte de manera breve las siguientes preguntas. 
¿Con qué parámetro del entrenamiento de redes neuronales se relaciona el step size empleado en los ataques adversarios?
¿Cuál es el objetivo de la función de loss propuesta por los autores de APGD en un ataque untargeted?



