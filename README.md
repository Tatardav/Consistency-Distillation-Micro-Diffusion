# Consistency Distillation — MicroDiT (EDM)

## Структура папки

| Файл | Назначение |
|------|------------|
| **testovoe.ipynb** | Основной ноутбук: подготовка данных, обучение, инференс, PartiPrompts бенчмарк, визуализации. Запускался в Kaggle. |
| **cd_utils.py** | Расписание σ, `edm_forward`, `cd_loss`. |
| **dataset.py** | `ImageCaptionDataset`, `InfiniteSampler`, `get_dataloader`. |
| **sampler.py** | `consistency_sample`, `generate_images`, загрузка промптов из TSV для бенчмарка. |
| **PartiPrompts.tsv** | Промпты для бенчмарка (колонка `Prompt`) |
| **prompts_to_generate.json** | Список текстовых промптов (случайных 5k из JourneyDB) для генерации обучающей выборки. |
| **requirements.txt** | Зависимости. |

Из репозитория **micro_diffusion** используется: `create_latent_diffusion`, DiT, VAE, текстовый энкодер, EDM config.

---

## Параметры окружения

- Ноутбук запускался в Kaggle с видеокартой NVIDIA H100 (ее можно выбрать при участии в [AI Mathematical Olympiad](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)). Репозитории [micro_diffusion](https://github.com/SonyResearch/micro_diffusion) и [Consistency-Distillation-Micro-Diffusion](https://github.com/Tatardav/Consistency-Distillation-Micro-Diffusion) добавлялись через Kaggle datasets.

---

## Краткая выжимка

### Тренировочный датасет

1. **Промпты:** из JourneyDB скачивается аннотация, из неё случайно выбирается 5000 промптов и сохраняются в `DATA_DIR/prompts_to_generate.json`.
2. **Картинки:** для каждого промпта из `prompts_to_generate.json` генерируется изображение с такими параметрами сэмплирования `num_inference_steps=30 guidance_scale=5.0, seed=2024`, картинки сохраняются в `DATA_DIR/images/`, аннотации — в `DATA_DIR/captions.json` (имя файла → промпт).

### Обучение

- **Учитель:** в качестве учителя использовалась модель [MicroDiT (dit_4_channel_0.5B_synthetic_data.pt)](https://huggingface.co/VSehwag24/MicroDiT/resolve/main/ckpts/dit_4_channel_0.5B_synthetic_data.pt).
- **Студент:** итоговый вариант студент обучался 10000 шагов, для валидации каждые 500 шагов сохранялись сгенерированные изображения и я вручную их просматривал

### Бенчмарк PartiPrompts

- **Данные:** файл **PartiPrompts.tsv** — таблица с колонкой `Prompt`. В коде читаются первые 200 строк .
- **Как сэмплируется:**  `generate_images(student_dit, model, batch_prompts, num_steps=4, seed=seed+i, device)`. Картинки сохраняются в `output_dir` (`output/parti_prompts_4steps/`).

### Результат

Примеры генерации студента (4 шага) по промптам из PDF:

![Примеры генерации по промптам из PDF](pdf_example_prompts_quality.png)

[Веса студента](https://drive.google.com/file/d/1ty-eR9Of6K90-WRRcx-Aag-mvHh-66_1/view?usp=sharing)

[Генерация по 200 первым промптам PartiPrompts](https://drive.google.com/file/d/1OnkCbCGB9akNsypKbqc1j0NtGDvM-T3S/view?usp=sharing)