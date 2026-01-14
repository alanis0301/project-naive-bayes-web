import base64
from django.shortcuts import render
from django.contrib import messages
from .utils import predictor

def index(request):
    """Lida com a exibição inicial e o processamento na mesma página"""
    prediction_result = None
    image_base64 = None
    
    if request.method == 'POST':
        if 'image' not in request.FILES:
            messages.error(request, 'Nenhuma imagem foi enviada.')
        else:
            image_file = request.FILES['image']
            
            if not image_file.content_type.startswith('image/'):
                messages.error(request, 'O arquivo enviado não é uma imagem válida.')
            else:
                try:
                    prediction_result = predictor.predict(image_file)
                    
                    image_file.seek(0)
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    image_base64 = f"data:{image_file.content_type};base64,{encoded_string}"
                    
                except Exception as e:
                    messages.error(request, f'Erro ao processar a imagem: {str(e)}')
    
    return render(request, 'classifier/index.html', {
        'prediction': prediction_result,
        'image_base64': image_base64
    })