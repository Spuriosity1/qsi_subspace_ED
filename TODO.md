1. Alter the pipeline a bit to introduce constraint-files, format
```json
{
    'version': 1,
    'max-id': 500:int
    'constraints': [
    {"indices": [0,8,16,24],
    "max-up": 2,
    "min-up": 2
    }
}
```

This allows for better tratment of the 111 polarised CSI
