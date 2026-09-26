o modelo está apresentando resultados péssimos. vamos fazer ajustes, primeiramente na lstm, para o desenvolvimento acelerado, e também na tcn. vou passar uma lista de ajustes, e você deverá fazer a testagem se vale a pena. itens marcados como obrigatórios deverão ser implementados independentemente.

volte a lstm e a lstm_bi para utilizarem como base a implementação antiga (obrigatório)
utilize apenas as features na implementação antiga, com exceção as que eu irei mandar adicionar (obrigatório)
adicione features que descreve o momento do dia, em componentes de seno e cosseno. ou seja. transforme o momento do dia (hora+minuto) em uma representação angular, e calcule o seno e o cosseno dela e utilize como feature
faça a função de ativação ser sigmoide
