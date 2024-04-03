import os
import glob
import plotly.graph_objects as go

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

def create_frame_from_obj(file_path):
    vertices, faces = read_obj(file_path)
    x, y, z = zip(*vertices)
    i, j, k = zip(*[(face[0]-1, face[1]-1, face[2]-1) for face in faces])
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='cyan', opacity=0.50)
    return go.Frame(data=[mesh])

# Carolina 디렉토리 내의 모든 OBJ 파일에 대해 작업 수행
carolina_directory = "./Carolina"
obj_files = glob.glob(os.path.join(carolina_directory, "*.obj"))
print(obj_files)

# 각 OBJ 파일로부터 Frame 생성
frames=[]
for i in range(1, 251):
    file_name = "Carolina/frame_{:04d}.obj".format(i)
    frames.append(create_frame_from_obj(file_name))
initial_path = "Carolina/frame_0001.obj"
vertices, faces = read_obj(initial_path)
x, y, z = zip(*vertices)
i, j, k = zip(*[(face[0]-1, face[1]-1, face[2]-1) for face in faces])
initial_mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='cyan', opacity=0.50)

fig = go.Figure(
    data=[initial_mesh],
    frames=frames,
    layout=go.Layout(
        scene=dict(
            # x, y, z 축의 격자(grid)와 제로 라인(zero line)을 숨깁니다.
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            # 원점의 좌표 위치를 고정하기 위한 축 범위 설정
        ),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 10, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}}],
                },
            ],
        }],
    )
)

fig.show()
