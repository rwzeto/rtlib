from PIL import Image
import time
import numpy as np
import sys
sys.path.append('target\\debug')
import rtlib as rt


def rot_mat_z(theta):
    return [[np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]]


def rot_mat_x(theta):
    return [[1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]]


def rot_mat_y(theta):
    return [[np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]]


def matmul(M1, M2):
    M1 = np.array(M1)
    M2 = np.array(M2)
    M3 = M1@M2
    return M3.tolist()


rot0 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
rot90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

cube_data = "data\\Cube_Tri.obj"
# cube2 = rt.poly3d(cube_data, index=1.5, pos=[0, 0, 300], rot=rot0, transmissivity=.5, reflectivity=0)
cube3 = rt.poly3d(cube_data, index=1.5, pos=[0, 0, 0], rot=matmul(rot_mat_y(20*np.pi/180), matmul(rot_mat_z(10*np.pi/180), rot_mat_x(20*np.pi/180))), transmissivity=.2, reflectivity=.8)
light = rt.light(pos=[-300, 0, 0], power=1, color=[0, 0, 0])

# pass rtlib a list of polyhedra

start = time.time()
scene = rt.scene()
# scene.add_obj(cube2)
scene.add_obj(cube3)
scene.add_light(light)


rowpixels, colpixels = (1000, 1000)
image = np.zeros((rowpixels, colpixels))

xmin, xmax = (-220, 220)
ymin, ymax = (-220, 220)

y = np.linspace(ymin, ymax, rowpixels)
x = np.linspace(xmin, xmax, colpixels)

start = time.time()
for row in range(np.shape(image)[0]):
    for col in range(np.shape(image)[1]):
        ray = rt.ray(index=1.0, pos=[y[col], x[row], -300], dir=[0, 0, 1], color=[1, 1, 1])
        image[row, col] = scene.render(ray)
        # print(image[row, col])
end = time.time()
print("%.2f s, %.2f px/s (%.2f fps)."%(end-start, (rowpixels*colpixels)/(end-start), 1/(end-start)))
image = image/np.max(image)*255


test_image = Image.fromarray(image.astype(np.uint8))
# test_image = test_image.convert('RGB')
test_image.save("outfile.png", format='PNG')
