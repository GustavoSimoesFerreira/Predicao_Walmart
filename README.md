# Predição Walmart

Este repositório contém um projeto de previsão de vendas para a Walmart, utilizando técnicas de machine learning.
O objetivo principal é prever as vendas futuras de lojas da Walmart com base em dados históricos e características das lojas.

## Descrição do Projeto

O projeto explora e utiliza um conjunto de dados fornecido pela Walmart no Kaggle.
O objetivo é construir um modelo preditivo que possa estimar as vendas futuras para cada loja e departamento.
O projeto foi feito em ambiente Google Colab (Jupyter Notebook).

## Estrutura do Repositório

- Predicao_Walmart/: Contém o Jupyter Notebook com a limpeza de dados, análise exploratória de dados (EDA), pré-processamento e modelagem.
	Também inclui todos os gráficos produzidos pelo script.

- Previsão_Vendas1.ipynb: Notebook principal com todos os passos do projeto.

- train.csv: Dados históricos de treinamento, que abrangem 2010–02–05 a 2012–11–01.

- stores.csv: Informações anônimas sobre as 45 lojas, indicando o tipo e tamanho da loja.

- features.csv: Dados adicionais relacionados às lojas, departamentos e atividade regional para as datas fornecidas.

- previsão_vendas.py: Script para todas análises.

- requisitos.txt: Lista de pacotes e suas versões necessárias para executar o projeto.

- README.md: Este arquivo com informações sobre o projeto.

## Como Rodar o Projeto

- Clone o repositório

	```bash
	git clone https://github.com/GustavoSimoesFerreira/Predicao_Walmart.git
	cd Predicao_Walmart
	```
	
- Instale as dependências

	```bash
	pip install -r requisitos.txt
	```

- Execute o notebook

	Abra o Jupyter Notebook e execute o notebook Previsão_Vendas1.ipynb para replicar a análise e modelagem.

- Resultados

	O notebook Previsão_Vendas1.ipynb apresenta as visualizações e análises detalhadas, bem como o desempenho dos modelos de previsão. As principais técnicas utilizadas incluem modelos de regressão e técnicas de machine learning avançadas.

- Contribuições
	
	Sinta-se à vontade para contribuir com o projeto! Se você encontrar problemas ou tiver sugestões, abra uma issue ou envie um pull request.

- Licença

	Este projeto está licenciado sob a Licença MIT.

- Referências

	CSVs do Walmart no Kaggle: kaggle.com/datasets/aslanahmedov/walmart-sales-forecast.

	Documentação dos pacotes utilizados pode ser encontrada em seus respectivos sites.
