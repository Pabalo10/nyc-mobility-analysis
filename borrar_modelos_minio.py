import os
import boto3
from dotenv import load_dotenv, find_dotenv

def exterminio_total_minio():
    load_dotenv(find_dotenv())
    
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")

    print("Conectando a MinIO (Modo Administrador)...")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # Buscamos tanto la carpeta con barra como sin barra por si acaso
    prefijo = f"{group_path}/modelos/"
    
    print(f"Buscando archivos ocultos y versiones antiguas en: {prefijo}")
    
    # Usamos un paginador especial para ver las versiones OCULTAS
    paginator = s3_client.get_paginator('list_object_versions')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefijo)
    
    archivos_destruidos = 0

    try:
        for page in page_iterator:
            # 1. Borrar las versiones reales de los archivos
            if 'Versions' in page:
                for version in page['Versions']:
                    s3_client.delete_object(
                        Bucket=bucket, 
                        Key=version['Key'], 
                        VersionId=version['VersionId'] # ¡Aquí está el truco!
                    )
                    archivos_destruidos += 1
                    if archivos_destruidos % 50 == 0:
                        print(f" - {archivos_destruidos} archivos reales destruidos...")

            # 2. Borrar las "Mantas de invisibilidad" (Delete Markers)
            if 'DeleteMarkers' in page:
                for marker in page['DeleteMarkers']:
                    s3_client.delete_object(
                        Bucket=bucket, 
                        Key=marker['Key'], 
                        VersionId=marker['VersionId']
                    )
                    archivos_destruidos += 1
                    if archivos_destruidos % 50 == 0:
                        print(f" - {archivos_destruidos} marcas de borrado destruidas...")

        # Por si quedó el "holograma" de la carpeta
        s3_client.delete_object(Bucket=bucket, Key=prefijo)
        s3_client.delete_object(Bucket=bucket, Key=f"{group_path}/modelos")

        if archivos_destruidos > 0:
            print(f"\n¡Exterminio absoluto completado! Se pulverizaron {archivos_destruidos} elementos ocultos.")
        else:
            print("\nNo se encontró nada. Si la carpeta sigue ahí, es un bug visual de la caché de tu navegador.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    exterminio_total_minio()