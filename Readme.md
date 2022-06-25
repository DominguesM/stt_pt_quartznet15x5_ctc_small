# Speech to Text - NeMo Portuguese

## Visão geral do modelo

Este modelo transcreve a fala em letras minúsculas do alfabeto português juntamente com espaços. É uma versão treinada com poucos exemplos do modelo QuartzNet-CTC.

## NVIDIA NeMo: Treinamento

Para treinar, ajustar ou brincar com o modelo, você precisará instalar o [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). Recomendamos que você o instale depois de instalar a versão mais recente do Pytorch.
```
pip instalar nemo_toolkit['all']
```

## Como usar este modelo

O modelo está disponível para uso no kit de ferramentas NeMo [1], e pode ser usado como um `pre-trained checkpoint` para inferência ou para ajuste fino em outro conjunto de dados.

### Instanciar automaticamente o modelo

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("dominguesm/stt_pt_quartznet15x5_ctc_small")
```

### ### Transcrição usando Python

Primeiro, vamos fazer o download de um pequeno exemplo:

```
wget https://github.com/DominguesM/stt_pt_quartznet15x5_ctc_small/raw/main/audios/common_voice_pt_25555332.mp3
```

Depois é só fazer:

```
asr_model.transcribe(['common_voice_pt_25555332.mp3'])
```

### Transcrever muitos arquivos de áudio

```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py  pretrained_name="dominguesm/stt_pt_quartznet15x5_ctc_small"  audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

### Input

Este modelo aceita áudios `mono-channel` de 16.000 KHz (arquivos wav) como entrada.

### Output

Este modelo fornece a fala transcrita como uma string para uma determinada amostra de áudio.

## Arquitetura do modelo

Esse modelo é baseado na arquitetura QuartzNet, que é uma variante do Jasper que usa `1D time-channel separable convolutional layers` em seus blocos residuais convolucionais e, portanto, são menores que os modelos Jasper.

Os modelos QuartzNet recebem segmentos de áudio e os transcrevem em sequências de letras, pares de bytes ou pedaços de palavras.

## Treinamento

O script com todos os passos relacionados ao treinamento deste modelo podem ser encontrados na pasta `notebooks`.

### Datasets

O modelo foi treinado com parte do conjunto de dados Common Voices 9.0 em português, totalizando 26 horas de áudio.

* Mozilla Common Voice (v9.0)

## Performance

| Métrica | Score |
| ------- | ----- |
| WER     | 49%   |
| CER     | 18%   |

As métricas foram obtidas utilizando o script abaixo:

**Atenção**: Os passos abaixo devem ser executados após o download do dataset (Mozilla Commom Voices 9.0 PT) e seguir os passos de pré-processamento dos dados de audio e arquivos `manifest` contidos no arquivo `notebooks/Finetuning CTC model Portuguese.ipynb`

```bash

$ wget -P scripts/ "https://raw.githubusercontent.com/NVIDIA/NeMo/v1.9.0/examples/asr/speech_to_text_eval.py"

$ wget -P scripts/ "https://raw.githubusercontent.com/NVIDIA/NeMo/v1.9.0/examples/asr/transcribe_speech.py"

$ python scripts/speech_to_text_eval.py \
    pretrained_name="dominguesm/stt_pt_quartznet15x5_ctc_small" \
    dataset_manifest="manifests/pt/commonvoice_test_manifest_processed.json" \
    output_filename="./evaluation_transcripts.json" \
    batch_size=32 \
    amp=true \
    use_cer=false

```


## Limitações

Como esse modelo foi treinado em conjuntos de dados de fala disponíveis publicamente, o desempenho desse modelo pode ser degradado para fala que inclui termos técnicos ou vernáculo em que o modelo não foi treinado. O modelo também pode ter um desempenho pior para a fala acentuada.

## Citação

Se você usar este trabalho, por favor, cite:


{% cite HuggingFace --file etc/citation.bib  %}

## References

[1] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)