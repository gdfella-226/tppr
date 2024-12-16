# LR1 Ранжировки

## Порядок запуска

### 0. Загрузка репозитория

```
git clone ssh://gitwork.ru/<project>
cd ./<project>
```

### 1. Установка зависимостей

```
cd ./LR2
pip install -r requirements.txt
```

### 1. Запуск

```
python ./main.py -f <file> -tp <trust_probe>
```
**Тестовые данные лежат в директории /\<project\>/data:**
* /data/l2.txt

**Доверительная вероятность trust_probe по умолчанию = 0.95**
