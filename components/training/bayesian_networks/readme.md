ERRORES QUE SE PUEDEN DAR:

-Los datos:
---- Tienen que tener una columna de ID y una de tiempo de medicion
- Por las variables:
----"id_column_name" es el id del objeto, "index_column_name" es el tiempo de medicion (por ejemplo el cycle index), tiene que venir dado en enteros
- Por el modelo :
----"discretization_quantiles" tiene que ser un entero o una lista de enteros
----funciona con datos categoricos
---- Si hay datos nulos, el modelo no funciona 
