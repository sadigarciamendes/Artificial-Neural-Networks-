# Primeiros neurônios (1943 a 1969)

## História

Ao longo da história, pesquisadores foram modelando propostas para a pesquisa na área. Em 1938, Rashevsky propôs um modelo matemático que usava equações diferenciais para descrever o comportamento neuronal (RASHEVSKY, 1996).

Em 1943, foi publicado o artigo *A Logical Calculus of the Ideas Immanent in Nervous Activity*, de McCulloch e Pitts. Eles propuseram um novo modelo matemático para descrever o neurônio, no qual ele está disparando ou inativo (1 ou 0).

Nos anos seguintes, houve mais estudos na área, como:

- o modelo de **Hodgkin e Huxley**;
- **Von Neumann**, investigando a capacidade de as Redes Neurais corrigirem seus próprios erros;
- **Hebb**, com o início do mecanismo de aprendizado.

### McCulloch e Pitts

O neurônio de McCulloch e Pitts (1943) é o primeiro modelo matemático de um neurônio e a base de todos os modelos seguintes. Ele funciona com três elementos:

- entradas binárias: cada entrada $x_i$ vale 0 ou 1;
- pesos fixos: cada entrada tem um peso $w_i$, que é $1$ se ela for excitatória ou $-1$ se for inibitória;
- limiar: um valor $\theta$ que a soma das entradas precisa atingir para o neurônio disparar.

O neurônio soma as entradas multiplicadas pelos pesos e compara o resultado com o limiar:

$$y = \begin{cases} 1, & \text{se } \sum_i w_i x_i \ge \theta \\ 0, & \text{caso contrário} \end{cases}$$

Escolhendo os pesos e o limiar, o mesmo neurônio reproduz portas lógicas:

| Porta | Pesos | Limiar | Dispara quando |
|---|---|---|---|
| AND | $1, 1$ | $2$ | as duas entradas são 1 |
| OR | $1, 1$ | $1$ | pelo menos uma entrada é 1 |
| NOT | $-1$ | $0$ | a entrada é 0 |

Como qualquer função lógica pode ser montada combinando essas portas, McCulloch e Pitts mostraram que redes desses neurônios são capazes de calcular qualquer expressão lógica.

A limitação do modelo é que não há aprendizado: os pesos e o limiar precisam ser definidos à mão para cada problema. Além disso, um neurônio sozinho só separa as saídas com uma reta, então ele não consegue representar o XOR.

> Esse exemplo está em `mcculloch_pitts.py`. A animação mostra o AND e o OR recebendo uma entrada por vez, com a soma e a saída de cada uma.

### Regra de Hebb

Em 1949, no livro *The Organization of Behavior*, Hebb propôs a primeira regra de aprendizado, utilizada até hoje em vários algoritmos:

> "Quando um estímulo de entrada influencia na produção de estímulos de saída, o peso da conexão entre os neurônios deve ser incrementado."

A conexão entre dois neurônios fica mais forte quando eles estão ativos ao mesmo tempo. Aplicada ao neurônio de McCulloch e Pitts, a regra resolve a limitação de os pesos serem definidos à mão. Para cada amostra, com entrada $x_i$ e saída desejada $d$:

$$w_i \leftarrow w_i + x_i \, d \qquad b \leftarrow b + d$$

Usando a representação bipolar ($-1$ e $1$), o peso aumenta quando a entrada e a saída têm o mesmo sinal e diminui quando têm sinais opostos. Os pesos começam em zero e basta uma passada pelas amostras.

Um exemplo com a porta AND:

| Amostra | $x_1$ | $x_2$ | $d$ | $w_1$ | $w_2$ | $b$ |
|---|---|---|---|---|---|---|
| início | | | | $0$ | $0$ | $0$ |
| 1 | $1$ | $1$ | $1$ | $1$ | $1$ | $1$ |
| 2 | $1$ | $-1$ | $-1$ | $0$ | $2$ | $0$ |
| 3 | $-1$ | $1$ | $-1$ | $1$ | $1$ | $-1$ |
| 4 | $-1$ | $-1$ | $-1$ | $2$ | $2$ | $-2$ |

No final, a reta $2x_1 + 2x_2 - 2 = 0$ separa corretamente as quatro entradas.

A limitação da regra é que ela não olha o erro: os pesos são reforçados sempre, mesmo quando o neurônio já acerta, e não há garantia de que o resultado final classifique tudo corretamente. É essa a diferença para o Perceptron, que só corrige os pesos quando erra.

> Esse exemplo está em `hebb.py`. A animação mostra a reta mudando a cada amostra apresentada, até separar o AND.

### Perceptron

Proposto por Rosenblatt em 1958, o Perceptron usa o mesmo neurônio de McCulloch e Pitts, mas os pesos não são mais definidos à mão, eles são aprendidos a partir dos exemplos.

O neurônio calcula a soma ponderada das entradas e passa o resultado por uma função degrau:

$$y = \begin{cases} 1, & \text{se } \sum_i w_i x_i + b \ge 0 \\ 0, & \text{caso contrário} \end{cases}$$

O aprendizado acontece amostra por amostra. A saída obtida $y$ é comparada com a saída desejada $d$, e os pesos são corrigidos pela regra:

$$w_i \leftarrow w_i + \eta \, (d - y) \, x_i \qquad b \leftarrow b + \eta \, (d - y)$$

em que $\eta$ é a taxa de aprendizado. Quando o neurônio acerta, o erro $(d - y)$ é zero e nada muda, os pesos só são alterados quando ele erra.

Geometricamente, o Perceptron traça uma reta (ou um hiperplano, com mais entradas) que divide o espaço em duas regiões. Se as classes forem linearmente separáveis, o algoritmo sempre converge. Se não forem, como no caso do XOR, ele nunca para de errar. Essa limitação foi apontada por Minsky e Papert em 1969.

> Esse exemplo está em `perceptron.py`. A animação mostra a reta se movendo a cada correção, com um círculo amarelo na amostra que causou o erro.

### Adaline

O Adaline (*Adaptive Linear Neuron*) foi proposto por Widrow e Hoff em 1960. A estrutura é a mesma do Perceptron, o que muda é onde o erro é medido.

No Perceptron, o erro é calculado depois da função degrau, então só existe "acertou" ou "errou". No Adaline, o erro é calculado na saída linear, antes do degrau:

$$s = \sum_i w_i x_i + b \qquad e = d - s$$

Com isso, o erro passa a ser um valor contínuo, e é possível definir uma função de custo, o erro quadrático médio:

$$E = \frac{1}{N} \sum_{n=1}^{N} (d_n - s_n)^2$$

Os pesos são ajustados na direção que diminui esse erro (descida do gradiente). Essa é a regra delta, também chamada de LMS ou regra de Widrow-Hoff:

$$w_i \leftarrow w_i + \eta \, (d - s) \, x_i \qquad b \leftarrow b + \eta \, (d - s)$$

As classes costumam ser representadas por $-1$ e $1$, e a função degrau só é aplicada no final, para classificar.

Na prática, a diferença para o Perceptron é que o Adaline corrige os pesos em toda amostra, proporcionalmente ao tamanho do erro, e continua ajustando a reta mesmo depois de classificar tudo corretamente. O resultado é uma fronteira mais centralizada entre as classes. A regra delta é a base do backpropagation usado nas redes de múltiplas camadas.

> Esse exemplo está em `adaline.py`. A animação mostra a fronteira à esquerda e a curva do erro quadrático médio caindo à direita.

### Madaline

O Madaline (*Many Adalines*), também de Widrow e Hoff, combina vários Adalines em uma mesma rede. Foi uma das primeiras redes com mais de uma camada, criada para resolver problemas que uma única reta não separa.

A estrutura usada aqui tem duas camadas:

- camada escondida: dois Adalines, cada um traçando a sua própria reta;
- saída: uma unidade com pesos fixos, que faz um OR entre os Adalines, ou seja, a rede dispara se pelo menos um deles disparar.

Com duas retas, a rede consegue resolver o XOR: cada Adaline isola um dos cantos da classe positiva, e o OR junta as duas regiões.

O treinamento usa a regra MRI (*Madaline Rule I*). Só os Adalines da camada escondida são treinados, e somente quando a rede erra:

- se a saída desejada é $1$ e nenhum Adaline disparou, corrige-se apenas o Adaline cuja soma estava mais perto de zero, isto é, o que estava mais perto de disparar;
- se a saída desejada é $-1$ e algum Adaline disparou, corrigem-se todos os que dispararam.

Em ambos os casos, a correção é a regra delta do Adaline. A ideia é fazer a menor mudança possível nos pesos para consertar o erro.

> Esse exemplo está em `madaline.py`. A animação mostra as duas retas se ajustando até separarem as quatro nuvens de pontos do XOR.

### Minsky e Papert

Em 1969, Minsky e Papert publicaram o livro *Perceptrons*, uma análise matemática do que o Perceptron consegue e não consegue fazer. O resultado mais conhecido é que um Perceptron de uma única camada só resolve problemas linearmente separáveis, e por isso não é capaz de representar funções simples como o XOR.

Já se sabia que redes com mais camadas poderiam resolver esses problemas, e o Madaline é um exemplo disso. O que faltava era um algoritmo geral para treinar as camadas escondidas: a regra MRI, por exemplo, só funciona porque a camada de saída é fixa.

O livro teve grande impacto. O interesse e o financiamento das pesquisas em redes neurais caíram bastante durante os anos 70, período que ficou conhecido como o "inverno" das redes neurais.

A área só voltou a crescer nos anos 80, principalmente com a popularização do algoritmo de *backpropagation* por Rumelhart, Hinton e Williams, em 1986. Ele permite treinar redes com várias camadas e é a base do Multi Layer Perceptron, o próximo tema deste repositório: [02-multi-layer-perceptron](../02-multi-layer-perceptron/).
