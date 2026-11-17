# Redes Neurais Artificiais

> "Redes Neurais Artificiais são redes massivamente paralelas e interconectadas, de elementos simples, organizadas de maneira hierárquica."
>
> (KOHONEN, 1972)

No entanto, há diversas definições sobre o que é uma Rede Neural Artificial. Todas partem do mesmo ponto, usar recursos de hardware e software para imitar as habilidades computacionais do sistema nervoso biológico (LOESCH, 1996).

Este é um estudo teórico e prático de Redes Neurais Artificiais. Neste repositório há tanto a explicação teórica quanto exemplos práticos, como guia de estudo.

## Sumário

O repositório está dividido nas seguintes partes, na ordem de estudo:

1. [Primeiros neurônios (1943 a 1969)](01-primeiros-neuronios/)
   - McCulloch e Pitts
   - Regra de Hebb
   - Perceptron
   - Adaline
   - Madaline
2. [Multi Layer Perceptron](02-multi-layer-perceptron/)
3. [Redes Neurais Recorrentes](03-redes-recorrentes/) (em construção)
4. [Redes Neurais Convolucionais](04-redes-convolucionais/) (em construção)

Cada pasta tem um `README.md` com a explicação teórica e os arquivos `.py` com os exemplos práticos.

## Como rodar

Os exemplos usam `numpy` e `matplotlib`. Entre na pasta e rode o arquivo, por exemplo:

```
cd 01-primeiros-neuronios
python perceptron.py
```

As animações abrem em uma janela. Para salvar em vídeo, descomente a linha `anim.save(...)` no fim do arquivo (precisa do `ffmpeg` instalado).
