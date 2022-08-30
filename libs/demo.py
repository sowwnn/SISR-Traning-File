import warnings
warnings.filterwarnings('ignore')
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt 
import base64
from io import BytesIO
import dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

###Layout
app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),

])
###Processing
def preprocess_b64(image_enc):
    """Preprocess b64 string into TF tensor"""
    decoded = base64.b64decode(str(image_enc).split("base64,")[-1])
    hr_image = tf.image.decode_image(decoded)

    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]

    return tf.expand_dims(tf.cast(hr_image, tf.float32), 0)


def tf_to_b64(tensor, ext="jpeg"):
    buffer = BytesIO()

    image = tf.cast(tf.clip_by_value(tensor[0], 0, 255), tf.uint8).numpy()
    Image.fromarray(image).save(buffer, format=ext)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{ext};base64, {encoded}"
###Model
def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value


model = tf.keras.models.load_model('/content/drive/MyDrive/mah_ws/ai_prj/prj_ImageSuper-Resolution/save_model/edsr/shuffle/200_baseline_x4_l1.h5',custom_objects={'PSNR':PSNR})
model.load_weights('/content/drive/MyDrive/mah_ws/ai_prj/prj_ImageSuper-Resolution/checkpoint/edsr/shuffle/200_baseline_x4_l1.h5')

##output
def parse_contents(contents,filename,sr_str):
    return html.Div([
        html.Div('Raw Content'),
        html.H5(filename),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Predict Content'),
        html.H5(filename),
        html.Img(src=sr_str),
        html.Hr(),
      

    ])

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def update_output(content,names):
    if content is None:
        return dash.no_update

    low_res = preprocess_b64(content)
    super_res = model(tf.cast(low_res, tf.float32))
    print(low_res)
    sr_str = str(tf_to_b64(low_res))
    children = [ parse_contents(c, n, sr) for c, n, sr in
            zip(content, names, sr_str)]
    return children

if __name__ == '__main__':
    app.run_server(debug=True)
