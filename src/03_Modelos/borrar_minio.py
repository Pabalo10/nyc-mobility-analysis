import os
import boto3
from botocore.client import Config
from dotenv import load_dotenv, find_dotenv

def auditoria_forense():
    load_dotenv(find_dotenv())
    
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")
    
    # Vamos a mirar la carpeta general "models" entera
    prefijo = f"{group_path}/models/"
    
    print(f"\n🔍 Auditando servidor real en: s3://{bucket}/{prefijo}")
    print("-" * 60)
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        paginas = paginator.paginate(Bucket=bucket, Prefix=prefijo)
        
        encontrados = 0
        for pagina in paginas:
            if 'Contents' in pagina:
                for obj in pagina['Contents']:
                    llave = obj['Key']
                    peso_kb = obj['Size'] / 1024
                    print(f" -> {llave} ({peso_kb:.2f} KB)")
                    encontrados += 1
                    
        print("-" * 60)
        print(f"Total de archivos reales en la base de datos: {encontrados}")
        
        if encontrados == 0:
            print("\n¡ATENCIÓN! La base de datos dice que está VACÍO.")
            print("Si sigues viendo la carpeta en la web, es 100% un FANTASMA VISUAL (caché del navegador o del servidor Nginx de la UCM).")
        
    except Exception as e:
        print(f"\n[X] Error leyendo la base de datos: {e}")

if __name__ == "__main__":
    auditoria_forense()