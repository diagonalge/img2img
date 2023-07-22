from fastapi.responses import StreamingResponse
import io
import os
import zipfile

def get_zip_response(output_folder_path):
    zip_bytes_io = io.BytesIO()
    with zipfile.ZipFile(zip_bytes_io, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for dirname, subdirs, files in os.walk(output_folder_path):
            zipped.write(dirname)
            for filename in files:
                zipped.write(os.path.join(dirname, filename))

    response = StreamingResponse(
                iter([zip_bytes_io.getvalue()]),
                media_type="application/x-zip-compressed",
                headers = {"Content-Disposition":f"attachment;filename=results.zip",
                            "Content-Length": str(zip_bytes_io.getbuffer().nbytes)}
            )
    return response