Temos duas séries temporais:
1. A série de setembro a novembro de 2021, que foi medida com LIDAR, utilizada no artigo original do Lucas
2. A série de novembro de 2021 até outubro de 2022, que foi medida com SODAR.

As duas séries temporais foram medidas no mesmo ponto, chamado de P1, com o equipamento LIDAR sendo substituído pelo SODAR

O SODAR possui uma qualidade de dados pior, por conta da leitura das informações depender da localização, já que quando o escoamento no sensor é muito homogêneo, ele não consegue realizar a estimativa. Além disso, ele é menos preciso nas leituras dos dados do que o LIDAR, já que o método de captura dele é mais instável e é afetado por condições climáticas locais, como vento e chuva. Além disso, o SODAR possui um alcance vertical menor do que o LIDAR.

As variáveis que são utilizadas são ws40, ws50, wdir40, v40, ws100, com ws100 sendo realmente o alvo de previsão, e as outras variáveis sendo transformadas e utilizadas como input. As features utilizadas são:
- ws100 (velocidade do vento a 100m)
- wsdisp40 (dispersão da velocidade do vento a 40m)
- vdisp40 (dispersão da componente vertical a 40m)
- dir40 (direção do vento a 40m)
- cis50 (cisalhamento do vento, (ws50-ws40)/(50-40))
Todas as features são suavizadas utilizando a Transformada de Wavelet