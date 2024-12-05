# Rede Neural Artificial

Este projeto foi desenvolvido para obtenção de nota na disciplina Aprendizado de Máquina (INE5638), ministrada pelo professor Eduardo Camilo Inacio, da Universidade Federal de Santa Catarina (UFSC).

O grupo é composto por:
* Bruno Garbatzki Madeira Cunha (19202601);
* Henrique Soares Carvalho (20200616);
* Manuela Schmitz (20102278).

A rede neural consiste no treinamento de três conjuntos de dados, sendo: um de regressão, um de classificação binária e um de classificação multiclasse.

---

Recomendamos a criação de um ambiente virtual antes de efetuar o download das bibliotecas:

```
python -m venv rna
rna\Scripts\activate

pip install -r requirements.txt
```


As bibliotecas utilizadas para desenvolvimento deste projeto são:

Para download dos datasets:

* kagglehub
* ucimlrepo

Para tratamento dos dados:

* numpy
* pandas
* matplotlib
* sklearn


Vale ressaltar que, no caso de utilização via Google Colab, é necessário apenas a instalação da ucimlrepo, da qual já está inclusa projeto. Segue link:

https://colab.research.google.com/drive/1u05ZDfpGY7dK8e2widIgpvcunn3-BjBs?usp=sharing


A execução deve ser realizada da seguinte forma:
```
python.exe src/binary.py
python.exe src/regression.py
python.exe src/multiclass.py
```