## Introdução

	O problema princpal enfrentado no projeto é a questão da predição do vento utilizando modelos de TCN e LSTM em uma estrutura de seq2seq. Para fazer isso, utilizamos um input de 12 horas (72 samples) de informações, para conseguirmos prever 6 horas (36 samples) no futuro.

## Dados

	Temos duas séries temporais principais, ambas medindo dados do vento em diferentes alturas, na mesma posição (chamada de P1), em um delta de 10 minutos por medição. Contudo, as duas são realmente uma série diferente por conta de serem utilizados instrumentos de medição diferentes, sendo o LIDAR utilizado durante o período de setembro de 2021 até novembro de 2021, e o SODAR utilizado durante o período de novembro de 2021 até outubro de 2022\.  
	A diferença entre ambos os instrumentos se dá pelo meio que é medido os dados do vento, com o LIDAR utilizando ondas luminosas, enquanto o SODAR utiliza ondas sonoras. O método de medição do SODAR acaba introduzindo inconsistências, já que o eco (tanto da chuva, como o da torre em si) distorce as leituras, temos distorção na medição da direção do vento, e o escoamento impede a leitura dos dados quando está muito estático. Como o escoamento estático impede a leitura dos dados, nós não temos uma série temporal contínua partindo de novembro de 2021, com vários gaps sendo presentes. Um trabalho que retrata bem as diferenças entre SODAR e LIDAR é “LIDAR and SODAR Measurements of Wind Speed and Direction in Upland Terrain for Wind Energy Purposes”  
	As informações medidas são: velocidade do vento, componente vertical do vento, direção do vento, componente zonal, componente , umidade, e temperatura. A série LIDAR mede de 40 até 260 metros, enquanto a série SODAR mede de 40 até 140 metros, com uma densidade maior de lacunas no dado em altitudes mais altas.  
	No entanto, apenas algumas variáveis são utilizadas, sendo:

* WS100 (velocidade do vento à 100m)  
* WSdisp40 (dispersão da velocidade do vento à 100m)  
* Vdisp40 (dispersão da componente vertical à 40m)  
* DIR40 (direção do vento a 40m)  
* CIS1 (cisalhamento entre 40m a 50m)

	Por fim, vale ressaltar a natureza em si do dado, com variações ao longo do tempo, tanto pela presença das diferentes temporadas, como pela mudança climática constante que é vista. A presença dessa variação ao longo do tempo, juntamente com a diferença dos instrumentos utilizados para a medição, exacerbada com o ambiente analisado, traz desafios com o domain shift e o time shift. Na localização de estudo, nós temos uma mudança climática muito impactante, onde setembro a novembro é o período de safra do vento, com chuvas fortes iniciando em dezembro e seguindo durante boa parte do ano, impactando a correlação das variáveis meteorológicas de ambos os períodos.

## Metodologia

	Para fazer a predição da velocidade do vento, é utilizado modelos de redes neurais, em uma estrutura de seq2seq com camada de atenção. As redes neurais utilizadas são a LSTM (com RNN), e a TCN (com CNN), e foram testados encoders bidirecionais para ambas redes. A arquitetura seq2seq normalmente é utilizada em problemas de linguagem natural, principalmente tradução, onde nós temos um encoder que recebe uma sequência de input, e gera um vetor de contexto, e o decoder lê esse vetor de contexto, e traduz ele para uma outra sequência.   
	Como a série SODAR possui diversas lacunas de dados pela natureza do instrumento, não é possível utilizá-la como dado de treinamento, o que impacta muito a generalização do modelo, pois a janela de treinamento fica muito dissonante com a janela operacional, trazendo uma necessidade de treinamento com dados recentes para termos um resultado aceitável. Por conta das lacunas da série SODAR, ela foi interpretada como uma coleção de diferentes séries temporais.  
	A metodologia dos testes feitos, então, segue a seguinte estratégia:

1. Normalização do dataset de treinamento (LIDAR) e cálculo das features relacionadas  
2. Treinamento do modelo  
3. Leitura das séries de teste (SODAR), e cálculo das features relacionadas  
4. Realização dos testes para cada série de teste (teste de predição geral, e teste de rolling forecast)  
5. Armazenamento dos resultados dos testes

	Na realização dos testes, cada série de teste é reinterpretada como várias janelas temporais de 18 horas (108 samples), com um avanço de uma amostra por janela. Na prática, isso se traduz na maior quantidade de predições para serem feitas, resultando em um teste mais rico de informação. Então, para cada janela de teste, é feito uma predição, utilizando 12 horas (72 samples) como input, e comparando o output predito com as outras 6 horas (36 samples) da janela, sendo esse resultado interpretado como o teste de predição geral.   
Então, para cada janela de predição realizada dentro de uma série de teste, é armazenada a última amostra predita, e posteriormente, é gerada uma série temporal com cada amostra predita armazenada, para realizar a comparação com a série de teste real, gerando assim o teste de rolling forecast. Esse teste é feito, pois a acurácia da predição de cada amostra é proporcional à sua posição dentro da predição, ou seja, a última amostra predita tende a possuir um erro maior do que a primeira amostra predita, já que a correlação semântica entre t e t-1 é muito maior quando comparada à t e t-36. 