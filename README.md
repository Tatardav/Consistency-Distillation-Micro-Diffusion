# Consistency Distillation — MicroDiT (EDM)

## Структура

```
cd_utils.py       — CD loss, EDM sigma schedule, EDM preconditioning
dataset.py        — ImageCaptionDataset + InfiniteSampler
sampler.py        — consistency sampling, PartiPrompts benchmark (из .tsv)
prepare_data.py   — скачивание JourneyDB subset
notebook.ipynb    — обучение + инференс + бенчмарк
PartiPrompts.tsv  — промпты для бенчмарка
```

## Быстрый старт

```bash
pip install -r requirements.txt
git clone https://github.com/SonyResearch/micro_diffusion.git
pip install -e micro_diffusion

# скачать чекпоинт teacher'а
wget https://huggingface.co/VSehwag24/MicroDiT/resolve/main/ckpts/dit_4_channel_0.5B_synthetic_data.pt```
```

## Алгоритм

CD адаптированная под EDM:

- Sigma schedule: степенной закон Karras, N=50 шагов
- Student получает зашумлённый latent `x_s = x_0 + σ_n · ε`, предсказывает `x_0`
- Teacher (с CFG) делает Euler ODE шаг: `x_t = x_s + (σ_{n+1} - σ_n) · (x_s - D_teacher) / σ_n`
- Target: тот же student на `(x_t, σ_{n+1})` с stop-gradient
- Loss: Huber между student prediction и target
- EDM preconditioning (`c_skip`, `c_out`) обеспечивает граничное условие `D(x; σ→0) ≈ x`
- Student не использует CFG — CFG-guided траектория учитедя дистиллирована в student

## Данные

3-5k изображений с подписями. В ноутбуке — автоматическое скачивание subset JourneyDB через HF Datasets (streaming).

Формат: `data_dir/images/*.jpg` + `data_dir/captions.json`

## Бенчмарк

PartiPrompts — первые 100 (или 400) промптов из `PartiPrompts.tsv`, генерация за 4 шага.
