# %% [markdown]
# # Practica Final
# Modulo 6 - Máster Data Science y Business Analytics
# Sergio Hervás Aragón

# %% [markdown]
# Importaciones

# %%
from pyspark.sql import SparkSession
import os
from pyspark.sql.functions import split, to_date, explode, round, mean, when, min, max, datediff, months_between

# %% [markdown]
# Al trabajar con pyspark, configuraremos el entorno creando una sesion de Spark

# %%
# master(String master): Establece la dirección URL maestra de Spark a la que se va a conectar
# appName(String name): Establece un nombre para la aplicación, que se mostrará en la interfaz de usuario web de Spark.
# config(String key, double value): Establece una opción de configuración.
# getOrCreate(): Obtiene una SparkSession existente o, si no hay ninguna, crea una nueva uno basado en las opciones establecidas en este constructor.
# Configuracion del nivel de registro de Spark para que no muestre los mensajes de advertencia
os.environ['PYSPARK_LOG_LEVEL'] = 'ERROR'
try:
    spark = SparkSession.builder\
        .master('local')\
        .appName('netflix_titles')\
        .config('spark.ui.port', '4050')\
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")\
        .getOrCreate()
except Exception as e:
    print(f'Ha ocurrido un error: {str(e)}')

# %% [markdown]
# 1.	Leer todos los csv descomprimidos guardados en la ruta de vuestro tmp en una sola línea de código (pista, usar wildcards para leer más de un fichero a la vez)

# %% [markdown]
# En primer lugar, procederemos a descarganos los archivos mediante una linea de comandos linux

# %%
# Declaro una lista con los archivos de los cuales voy ha hacer uso y me voy a descargar
csv_files = [
    '/netflix_titles_dirty_01.csv',
    '/netflix_titles_dirty_02.csv',
    '/netflix_titles_dirty_03.csv',
    '/netflix_titles_dirty_04.csv',
    '/netflix_titles_dirty_05.csv',
    '/netflix_titles_dirty_06.csv',
    '/netflix_titles_dirty_07.csv'
]

# Variable la cual usaremos en un futuro para indicar donde almacenar los archivos de la lista
first_ending = '/tmp'
# Variable la cual usaremos en un futuro para indicar la manera en la que nos vamos a descargar los archivos de la lista
ultimate_termination = '.gz'
# Bucle con el que recorreremos la lista
for file in csv_files:    
    # Declararemos dos variables que nos serviran para validar, si: 1º - Esta descargado el archivo, y 2º - Si el archivo esta descomprimido
    compressed_path = first_ending + file + ultimate_termination
    decompressed_path = first_ending + file    
    # Buscamos el path, y si no existe tanto el comprimido como el descomprimido, lo descargaremos y los descomprimiremos
    if not os.path.exists(compressed_path and decompressed_path):
        url = f'https://github.com/datacamp/data-cleaning-with-pyspark-live-training/blob/master/data{file}.gz?raw=True'
        try:
            # Descargamos
            ! wget -O $compressed_path $url
            # Descomprimimos
            ! gunzip $compressed_path
        except Exception as e:
            print(f'Ha ocurrido un error: {str(e)}')
        finally:
            print(f'Archivo {file} descargado y descomprimido ')
    else:
        print(f'El fichero {file} ya existe')

# %%
df = spark.read.csv('./../../../tmp/*csv', sep='\t', header=False)
df.show(1, truncate=False)

# %% [markdown]
# 2.	Analiza las columnas y renómbralos con un nombre que tenga sentido para cada una

# %%
# Declaro una lista que contendran los nombres de las columnas nuevas las cuales van a ser las que van a dar nombre a cada columna
columsNames = ['id','type','movie_name','director','actors','country','release_date','year','age_classification','duration','gender','description']

for item in range(len(columsNames)):
    # Reemplazo con el bucle, en primera posicion, el nombre de la columba original, y en segunda posicion el nuevo nombre
    df = df.withColumnRenamed(f'_c{item}', columsNames[item])
    
df.show(1)

# %% [markdown]
# 3.	Limpia el dataframe para que no existan nulos, adicionalmente elimina todos los valores que no se correspondan con el resto de datos de la columna

# %%
# Declaremos un metodo que nos servira para ver cuantos nulos hay por columnas en nuestro df
def counts_off_nulls_spark(columsNames, df):
    print('Conteo de nulos por columnas:')
    for columns_name in columsNames:
        # Recorro el array de mis columnas filtrando por nombre de las columnas los valores que son nulos, y haciendo un conteo de estos
        number_nulls_columns = df.filter(df[columns_name].isNull()).count()
        print(f'\t{columns_name}: {number_nulls_columns} nulls number')
    

# %%
# Funcion que nos realiza un conteo de nulos antes de eliminarlos
counts_off_nulls_spark(columsNames,df)
# Eliminamos los nulos
df = df.dropna()
# Funcion que nos realiza un conteo de nulos despues de eliminarlos
counts_off_nulls_spark(columsNames,df)

# %% [markdown]
# 4.	Revisa el tipo de dato de cada columna y parsealo según corresponda (la columna duración debe ser numérico)

# %%
try:
    # Realizamos primeramente un arreglo de la columna duracion, donde separaremos el contenido mediente el espacio formando dos
    # columnas con la respectiva información, y elimaremos la columna original
    if 'duration' in df.columns:
        df = df.\
            withColumn('time_duration', split(df['duration'], ' ')[0]).\
            withColumn('type_duration', split(df['duration'], ' ')[1]).\
            drop('duration')
except Exception as e:
    print(f'Ha ocurrido un error: {str(e)}')
finally:
    print(f'Parseo de fecha finalizado')
    
try:
# Una vez todas las columnas como queremos, procedemos al parseo de la información, emprezando por los enteros (id, year, time_duration)
    df = df.\
        withColumn('id', df['id'].cast('int')).\
        withColumn('year', df['year'].cast('int')).\
        withColumn('time_duration', df['time_duration'].cast('int'))
except Exception as e:
    print(f'Ha ocurrido un error: {str(e)}')
finally:
    print(f'Parseo de enteros finalizado')
    
try:
    # Seguidamente procederemos al parseo de la información en tipo Date (release_date)
    df = df.withColumn("release_date", to_date(df['release_date'], 'MMMM dd, yyyy'))
except Exception as e:
    print(f'Ha ocurrido un error: {str(e)}')
finally:
    print(f'Parseo de fechas finalizado')
    
df.printSchema()
df.show(truncate=False)

# %% [markdown]
# 5.	Calcula la duración media en función del país

# %%
# Previo al tratamiento de la información, realizamos una visualización de los elementos unicos por columna.
unique_values_of_type_duration = df.select('type_duration').distinct()
unique_values_of_country = df.select('country').distinct()
# unique_values_of_type_duration.show(truncate=False)
# unique_values_of_country.show(truncate=False)

# Tras visualizar que hay varias ciudades por registro, las separamos (explide) entre si reproduciendo sus valores por cada registro
unique_countries = df.withColumn('country', explode(split(df['country'], ', ')).alias('unique_countries'))
# unique_countries.show()

# Realizamos la duracion media en funcion de cada pais, donde agruparemos por temporadas y minutos
average_duration_by_country = unique_countries.groupBy('country').agg(\
    # redondeo de 2 decimales (round), de la media (mean), cuando tenemos el tipo que queremos (when)
    round(mean(when((df['type_duration'] == 'Season') | (df['type_duration'] == 'Seasons'), df['time_duration'])), 2).alias('season_average'),\
    round(mean(when(df['type_duration'] == 'min', df['time_duration'])), 2).alias('movie_stocking'))
average_duration_by_country.show()



# %% [markdown]
# 6.	Filtra las películas y series que contengan la palabra music en su descripción y que su duración sea mayor a 90 minutos, ¿cuál es el actor que más películas y series ha realizado bajo estas condiciones?

# %%
leak = df.filter((df['description'].like('%music%')) & (df['time_duration'] > 90))
# leak.show(truncate=False)

separation_of_actors = leak.withColumn('actors', explode(split(df['actors'], ', ')))

filtered_by_actors = separation_of_actors.groupBy('actors').count().orderBy('count', ascending=False)
# filtered_by_actors.show(1, truncate=False)

print('El actor que cumple las condiciones es', filtered_by_actors.first()['actors'])

# %% [markdown]
# 7.	Para el actor que más producciones ha realizado calcula cuantas semanas han pasado desde su primera producción hasta su última.

# %%
# Divido lod actores sin nunguna restricción 
separation_of_actors_without_conditions = df.withColumn('actors', explode(split(df['actors'], ', ')))
# Agrupo los actores y realizo un conteo, el cual ordeno de manera ascendente
actor_with_more_productions = separation_of_actors_without_conditions.groupBy('actors').count().orderBy('count', ascending=False)
# Obtengo el actor deseado
selected_actor = actor_with_more_productions.first()['actors']
# A la hora de coger las fechas, buscamos la fila que contiene nuestro actor, y obtenemos de todos los registros donde sale,
# el valor minimo y el valor maximo
date_range = separation_of_actors_without_conditions.\
    filter(separation_of_actors_without_conditions['actors'] == selected_actor).\
    select(min('release_date'), max('release_date')).first()

# Sacamos los dias de diferencia entre la fecha máxima y la mínima
days_difference = date_range[1] - date_range[0]
# Y divido el resultado entre 7 para recuperar las semanas
weeks_between = days_difference/7
print(f'De primera producción hasta su última de {selected_actor} han pasado {weeks_between.days} semanas')

# %% [markdown]
# 8.	Transforma la columna de géneros para que su contenido sea un array con los valores de cada género por registro

# %%
df = df.withColumn('gender', split(df['gender'], ', '))

# %% [markdown]
# 9.	¿Cuántas producciones se han realizado en un único país y cuantas tienen 2 o más países?

# %%
# Divido los paises sin nunguna restricción 
separate_countries = df.withColumn('country', explode(split(df['country'], ', ')))

# Hacemos un conteo por cada vez que un pais aparece
production_count = separate_countries.groupBy("country").count()
# Seguidamente realizaremos la distincion, donde pondremos en una columna los que aparezcan en una, y en otra los que aparezcan más de una vez
differentiation = production_count.\
    withColumn('countries with one registration', when(production_count['count'] == 1, production_count['country'])).\
    withColumn('countries with more than one registration', when(production_count['count'] > 1, production_count['country']))
    
# Me quedo con las 2 columnas en cuestion, donde se puede visualizar lo filtrado
differentiation = differentiation.\
    select('countries with one registration','countries with more than one registration')

differentiation.show()

# %% [markdown]
# 10.	Escribe el dataframe final como un fichero parquet

# %%
df.write.format('parquet').mode('overwrite').save('./../../../tmp/parquet/')

# %%
spark.stop()

# %% [markdown]
# # Bibliografía 
# 
#  - [Como hacer hipervinculos](https://learn.microsoft.com/es-es/contribute/content/how-to-write-links)
# 
# #### Apartado 1:
# 
#  - [¿Que son los wildcards?](https://support.microsoft.com/en-us/office/examples-of-wildcard-characters-939e153f-bd30-47e4-a763-61897c87b3f4#:~:text=Wildcards%20are%20special%20characters%20that,named%20John%20on%20Park%20Street.)
# 
#  - [Visualización de la ejecución de mi aplicación netflix_titles en modo local (http://localhost:4050/)](http://localhost:4050/)
# 
#  - [Información del objeto builder](https://spark.apache.org/docs/3.2.0/api/java/org/apache/spark/sql/SparkSession.Builder.html)
# 
#  - [Expansión del patrón de nombres de ruta de estilo Unix (glob)](https://docs.python.org/es/3/library/glob.html)
# 
#  - [path](https://www.guru99.com/es/python-check-if-file-exists.html)
# 
#  - [Ejecutar comandos de shell en Jupyter Notebook](https://blogs.upm.es/estudiaciencia/variables-en-bash/)
# 
#  #### Apartado 2:
# 
#  - [Cambiar el nombre de columnas usando 'withColumnRenamed'](https://www.machinelearningplus.com/pyspark/pyspark-rename-columns/?utm_content=cmp-true)
# 
#  #### Apartado 3:
# 
#  - [pyspark.sql.DataFrame.printSchema](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.printSchema.html)
# 
#  - [truncate (referencia apartado 11)](https://stackoverflow.com/questions/33742895/how-to-show-full-column-content-in-a-spark-dataframe)
# 
#  #### Apartado 4:
# 
#  - [Cast Column Type With Example](https://sparkbyexamples.com/pyspark/pyspark-cast-column-type/)
# 
#  - [Spark – Split DataFrame single column into multiple columns](https://sparkbyexamples.com/spark/spark-split-dataframe-column-into-multiple-columns/)
# 
#  - [Ver etiquetas de columnas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html)
# 
#  - [INCONSISTENT_BEHAVIOR_CROSS_VERSION.PARSE_DATETIME_BY_NEW_PARSER](https://community.databricks.com/t5/data-engineering/inconsistent-behavior-cross-version-parse-datetime-by-new-parser/td-p/43674)
# 
#  #### Apartado 5
# 
#  - [PySpark Explode Array and Map Columns to Rows](https://sparkbyexamples.com/pyspark/pyspark-explode-array-and-map-columns-to-rows/)
# 
#  - [Round Column Values](https://www.statology.org/pyspark-round-to-2-decimal-places/)
# 
#  - [Conditional when](https://pratikbarjatya.medium.com/mastering-pyspark-when-statement-a-comprehensive-guide-691c1f14a597)
# 
#  #### Apartado 6
# 
#  - [Filter](https://sparkbyexamples.com/pyspark/pyspark-where-filter/)
# 
#  - [pyspark.sql.DataFrame.orderBy](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.orderBy.html)
# 
#  - [First()](https://aitor-medrano.github.io/iabd2223/spark/02dataframeAPI.html#mostrando-los-datos)
# 
#  #### Apartado 7
# 
#  - [pyspark.sql.functions.min](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.min.html)
# 
#  - [PySpark – Difference between two dates (days, months, years)](https://sparkbyexamples.com/pyspark/pyspark-difference-between-two-dates-days-months-years/)


