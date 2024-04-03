from flask import Flask, render_template, send_from_directory, request
import os
import plotly.graph_objects as go
import plotly
import json
import os
import glob
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# 폴더 내 파일 목록을 반환하는 루트
@app.route('/')
def list_files():
    files = os.listdir('static/animations')
    files = sorted(files)
    return render_template('index.html', files=files)

# 특정 파일을 디스플레이하는 루트
@app.route('/animations/<filename>')
def show_animation(filename):
    carolina_directory = os.path.join('static/animations', filename)
    obj_files = sorted(glob.glob(os.path.join(carolina_directory, "*.obj")))

    # 각 OBJ 파일로부터 Frame 생성
    frames=[]
    for file_name in obj_files:
        frames.append(create_frame_from_obj(file_name))
    initial_path = obj_files[0]
    initial_mesh = create_frame_from_obj(initial_path).data[0]

    fig = go.Figure(
        data=[initial_mesh],
        frames=frames,
        layout=go.Layout(
            scene=dict(
                # 여기에 원점의 좌표 위치를 고정하기 위한 축 범위 설정 등을 추가
                aspectmode='data',
            ),
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True, "mode": "immediate"}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }],
        )
    )

    # 슬라이더 추가
    sliders = [{
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }]

    # 슬라이더의 각 스텝을 frames에 맞춰 구성
    for i, frame in enumerate(frames):
        slider_step = {"args": [
            [frame.name],
            {"frame": {"duration": 80, "redraw": True}, "mode": "immediate", "transition": {"duration": 100}}
        ],
            "label": str(i),
            "method": "animate"}
        sliders[0]["steps"].append(slider_step)

    fig.update_layout(sliders=sliders)

    html_str = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    # # 간단한 Plotly 3D 스캐터 플롯 생성
    # fig = go.Figure(data=[go.Scatter3d(x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3],
    #                                    mode='markers')])
    # html_str = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')

    return html_str






# for plotly 3d visulaizeation functions
def create_frame_from_obj(file_path):
    vertices, faces = read_obj(file_path)
    x, y, z = zip(*vertices)
    
    if faces:  # 면 데이터가 있는 경우
        i, j, k = zip(*[(face[0]-1, face[1]-1, face[2]-1) for face in faces])
        mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='cyan', opacity=0.50)
        return go.Frame(data=[mesh])
    else:  # 면 데이터가 없는 경우, 점으로 표시
        points = go.Scatter3d(x=x, y=y, z=z, mode='markers', 
                              marker=dict(size=2, color='cyan', opacity=0.5))
        return go.Frame(data=[points], name=str(file_path))


def read_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # 정점
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('f '):  # 면
                face = [int(face.split('/')[0]) for face in line.strip().split()[1:]]
                faces.append(face)
    return vertices, faces






if __name__ == '__main__':
    app.run(debug=True)