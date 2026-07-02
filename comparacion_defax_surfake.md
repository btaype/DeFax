# Comparacion tecnica DeFaX vs SurFake

> Nota: los archivos de pesos con extensión .pth se ignoran automáticamente por Git en cualquier subcarpeta del proyecto mediante la regla definida en [.gitignore](.gitignore).

Esta comparacion se basa en las metricas obtenidas localmente para ambos modelos. Para DeFaX se usaron los archivos `checkpoints/metricas_test_original.json` y `checkpoints/metricas_test_externo.json`. Para SurFake se usaron `resultados_test_split_train.csv`, `resultados_test.csv` y las matrices de confusion `matriz_confusion_test_split_train.png` y `matriz_confusion_test.png`.

## Resultados sobre FaceForensics++

| Modelo | Accuracy | Precision | Recall | F1-score | AUC-ROC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SurFake | 0.978 | 0.992 | 0.981 | 0.986 | 0.996 | 3221 | 139 | 312 | 16470 |
| DeFaX | 0.902 | 0.992 | 0.889 | 0.938 | 0.977 | 3235 | 125 | 1855 | 14927 |

En FaceForensics++, SurFake obtuvo el mejor desempeno global. Supero a DeFaX en accuracy por 0.076, en F1-score por 0.049 y en AUC-ROC por 0.019. La diferencia mas importante esta en el recall de la clase fake: SurFake alcanzo 0.981, mientras que DeFaX obtuvo 0.889. Esto significa que SurFake dejo pasar muchas menos imagenes falsas como reales: 312 falsos negativos frente a 1855 en DeFaX.

DeFaX, sin embargo, fue ligeramente mejor al evitar falsos positivos sobre imagenes reales: tuvo 125 FP, frente a 139 FP de SurFake. Por eso la precision de ambos modelos fue practicamente igual, cercana a 0.992. En terminos practicos, DeFaX fue muy conservador al predecir fake: cuando predijo fake casi siempre acerto, pero no detecto tantas muestras fake como SurFake.

La razon tecnica principal es que SurFake esta mas alineado con FaceForensics++. Su entrada RGB+GSD explota inconsistencias geometricas de superficie facial, que aparecen de forma consistente en manipulaciones como DeepFakes, Face2Face, FaceSwap, FaceShifter y NeuralTextures. Al usar el descriptor GSD junto con RGB, MobileNetV2 recibe una senal adicional directamente relacionada con deformaciones faciales. En este conjunto, esa senal parece ser altamente discriminativa.

DeFaX, por otro lado, trabaja solo con RGB y fusiona caracteristicas locales de EfficientNet con caracteristicas globales de Swin Transformer mediante cross-attention. Esta arquitectura es mas rica, pero tambien mas pesada y fue originalmente propuesta para rostros generados sintetica o completamente por IA. Al evaluarla sobre manipulaciones faciales de FaceForensics++, su precision fue muy alta, pero su recall menor indica que algunas manipulaciones no activaron suficientemente los patrones aprendidos por el modelo.

## Resultados sobre WildDeepfake / dataset externo

| Modelo | Accuracy | Precision | Recall | F1-score | AUC-ROC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeFaX | 0.649 | 0.690 | 0.546 | 0.609 | 0.706 | 2536 | 834 | 1544 | 1854 |
| SurFake | 0.560 | 0.602 | 0.363 | 0.453 | 0.598 | 2555 | 815 | 2165 | 1233 |

En el dataset externo, DeFaX fue el mejor modelo. Supero a SurFake en accuracy por 0.089, en precision por 0.088, en recall por 0.183, en F1-score por 0.156 y en AUC-ROC por 0.108. La diferencia mas relevante vuelve a estar en la deteccion de imagenes falsas: DeFaX detecto 1854 falsos correctamente, mientras que SurFake solo detecto 1233. En consecuencia, SurFake produjo 2165 falsos negativos, frente a 1544 de DeFaX.

SurFake fue apenas mejor identificando imagenes reales: obtuvo 2555 TN y 815 FP, mientras que DeFaX obtuvo 2536 TN y 834 FP. Sin embargo, esta ventaja es pequena y no compensa la gran perdida de recall sobre la clase fake. Para un detector de deepfakes, dejar pasar imagenes falsas como reales es un error critico, por lo que en el escenario externo DeFaX es la opcion mas robusta.

La caida de SurFake se explica por cambio de dominio. El GSD funciona muy bien cuando las inconsistencias geometricas del dataset externo se parecen a las del entrenamiento. Pero en WildDeepfake las imagenes pueden presentar variaciones mas reales de compresion, iluminacion, pose, resolucion, recorte, ruido y origen visual. En ese contexto, la geometria extraida por GSD puede dejar de ser una senal estable o puede no reflejar las mismas anomalias aprendidas en FaceForensics++.

DeFaX tambien cae en el dataset externo, pero conserva mejor la separacion entre clases. Su AUC-ROC sube a 0.706 frente al 0.598 de SurFake, lo que indica que sus probabilidades aun contienen mas informacion discriminativa. Esto puede deberse a que la fusion EfficientNet + Swin permite capturar tanto texturas locales como relaciones globales del rostro, haciendo que el modelo dependa menos de una unica representacion geometrica.

## Comparacion de generalizacion

| Modelo | Drop Accuracy | Drop F1-score | Drop AUC-ROC |
|---|---:|---:|---:|
| DeFaX | 0.253 | 0.329 | 0.271 |
| SurFake | 0.418 | 0.534 | 0.398 |

Ambos modelos pierden rendimiento al pasar de FaceForensics++ a WildDeepfake, lo que confirma que la generalizacion sigue siendo el principal problema en deteccion de deepfakes. No obstante, la perdida de SurFake fue mucho mayor. Su F1-score bajo de 0.986 a 0.453, mientras que DeFaX bajo de 0.938 a 0.609. Por tanto, SurFake es superior dentro del dominio de entrenamiento, pero DeFaX generaliza mejor fuera de ese dominio.

## Conclusion comparativa

SurFake fue el mejor modelo en FaceForensics++, especialmente por su alto recall sobre la clase fake y su bajo numero de falsos negativos. Esto indica que el uso de RGB+GSD es muy efectivo cuando el dataset contiene manipulaciones faciales con inconsistencias geometricas similares a las vistas durante el entrenamiento.

DeFaX fue el mejor modelo en el dataset externo WildDeepfake. Aunque no alcanza el rendimiento de SurFake en FaceForensics++, su caida fuera del dominio fue menor y mantuvo mejores valores de accuracy, precision, recall, F1-score y AUC-ROC. Esto sugiere que la fusion por cross-attention entre caracteristicas locales y globales ofrece una representacion mas flexible ante datos no observados.

En terminos practicos, SurFake es preferible si el escenario de prueba es similar a FaceForensics++ y se dispone de mapas GSD confiables. DeFaX es preferible cuando interesa una mayor robustez frente a datasets externos o imagenes provenientes de distribuciones diferentes. La comparacion tambien muestra que una posible linea de mejora seria combinar ambas ideas: usar la informacion geometrica GSD de SurFake, pero fusionarla a nivel de caracteristicas mediante un mecanismo de atencion cruzada similar al de DeFaX, en lugar de concatenarla directamente como canales de entrada.

## Ajuste recomendado para el PDF

La conclusion del PDF aun dice que SurFake y DeFaX "se implementaran y compararan" como siguiente paso. Esa frase deberia cambiarse porque el documento ya presenta resultados experimentales. Una version mas consistente seria:

"En este trabajo se implementaron y compararon SurFake y DeFaX bajo el mismo dataset FaceForensics++ y tambien sobre un conjunto externo WildDeepfake. Los resultados muestran que SurFake obtiene el mejor rendimiento dentro del dominio de entrenamiento, alcanzando 0.978 de accuracy y 0.996 de AUC-ROC en FaceForensics++. Sin embargo, en el dataset externo su rendimiento disminuye considerablemente. DeFaX, aunque obtiene resultados menores en FaceForensics++, presenta mejor comportamiento fuera del dominio, superando a SurFake en WildDeepfake con 0.649 de accuracy y 0.706 de AUC-ROC. Esto evidencia que la generalizacion sigue siendo el desafio principal en deteccion de deepfakes y que los resultados deben analizarse no solo en el dataset original, sino tambien en escenarios externos."
