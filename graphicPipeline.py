import numpy as np
from mipmap import (build_mipmaps, sample_nearest, sample_bilinear,
                    sample_trilinear, sample_anisotropic,
                    compute_lod_accurate)


class Fragment:
    def __init__(self, x : int, y : int, depth : float, interpolated_data,
                 lod=0.0, dudx=0.0, dvdx=0.0, dudy=0.0, dvdy=0.0):
        self.x = x
        self.y = y
        self.depth = depth
        self.interpolated_data = interpolated_data
        self.lod = lod
        self.dudx = dudx
        self.dvdx = dvdx
        self.dudy = dudy
        self.dvdy = dvdy
        self.output = []

def edgeSide(p, v0, v1) :
    return (p[0]-v0[0])*(v1[1]-v0[1]) - (p[1]-v0[1])*(v1[0]-v0[0])

class GraphicPipeline:
    def __init__ (self, width, height, filter_mode="trilinear", downsample_filter="box"):
        self.width = width
        self.height = height
        self.filter_mode = filter_mode
        self.downsample_filter = downsample_filter
        self.image = np.zeros((height, width, 3))
        self.depthBuffer = np.ones((height, width))
        self.mips = None


    def VertexShader(self, vertex, data) :
        outputVertex = np.zeros((18))

        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        w = 1.0

        vec = np.array([[x],[y],[z],[w]])

        vec = np.matmul(data['projMatrix'],np.matmul(data['viewMatrix'],vec))

        outputVertex[0] = vec[0]/vec[3]
        outputVertex[1] = vec[1]/vec[3]
        outputVertex[2] = vec[2]/vec[3]

        outputVertex[3] = vertex[3]
        outputVertex[4] = vertex[4]
        outputVertex[5] = vertex[5]

        outputVertex[6] = data['cameraPosition'][0] - vertex[0]
        outputVertex[7] = data['cameraPosition'][1] - vertex[1]
        outputVertex[8] = data['cameraPosition'][2] - vertex[2]

        outputVertex[9] = data['lightPosition'][0] - vertex[0]
        outputVertex[10] = data['lightPosition'][1] - vertex[1]
        outputVertex[11] = data['lightPosition'][2] - vertex[2]

        outputVertex[12] = vertex[6]
        outputVertex[13] = vertex[7]

        outputVertex[14] = outputVertex[0]
        outputVertex[15] = outputVertex[1]

        outputVertex[16] = (outputVertex[0] + 1.0) * 0.5 * self.width
        outputVertex[17] = (outputVertex[1] + 1.0) * 0.5 * self.height

        return outputVertex


    def Rasterizer(self, v0, v1, v2) :
        fragments = []

        #culling back face
        area = edgeSide(v0,v1,v2)
        if area <= 0 :
            return fragments


        #AABBox computation
        #compute vertex coordinates in screen space
        v0_image = np.array([0,0])
        v0_image[0] = (v0[0]+1.0)/2.0 * self.width
        v0_image[1] = ((v0[1]+1.0)/2.0) * self.height

        v1_image = np.array([0,0])
        v1_image[0] = (v1[0]+1.0)/2.0 * self.width
        v1_image[1] = ((v1[1]+1.0)/2.0) * self.height

        v2_image = np.array([0,0])
        v2_image[0] = (v2[0]+1.0)/2.0 * self.width
        v2_image[1] = (v2[1]+1.0)/2.0 * self.height

        #compute the two point forming the AABBox
        A = np.min(np.array([v0_image,v1_image,v2_image]), axis = 0)
        B = np.max(np.array([v0_image,v1_image,v2_image]), axis = 0)

        #cliping the bounding box with the borders of the image
        max_image = np.array([self.width-1,self.height-1])
        min_image = np.array([0.0,0.0])

        A  = np.max(np.array([A,min_image]),axis = 0)
        B  = np.min(np.array([B,max_image]),axis = 0)

        #cast bounding box to int
        A = A.astype(int)
        B = B.astype(int)
        #Compensate rounding of int cast
        B = B + 1

        u0, v_0 = v0[12], v0[13]
        u1, v_1 = v1[12], v1[13]
        u2, v_2 = v2[12], v2[13]

        dx_01 = v1_image[0] - v0_image[0]; dy_01 = v1_image[1] - v0_image[1]
        dx_02 = v2_image[0] - v0_image[0]; dy_02 = v2_image[1] - v0_image[1]
        du_01 = u1 - u0;                   dv_01 = v_1 - v_0
        du_02 = u2 - u0;                   dv_02 = v_2 - v_0

        det = dx_01 * dy_02 - dx_02 * dy_01
        if abs(det) > 1e-10:
            inv_det = 1.0 / det
            dudx = (du_01 * dy_02 - du_02 * dy_01) * inv_det
            dudy = (dx_01 * du_02 - dx_02 * du_01) * inv_det
            dvdx = (dv_01 * dy_02 - dv_02 * dy_01) * inv_det
            dvdy = (dx_01 * dv_02 - dx_02 * dv_01) * inv_det
        else:
            dudx = dvdx = dudy = dvdy = 0.0

        if self.mips is not None:
            tex_size = max(self.mips[0].shape[0], self.mips[0].shape[1])
        else:
            tex_size = 1

        lod = compute_lod_accurate(dudx, dvdx, dudy, dvdy, tex_size)

        #for each pixel in the bounding box
        for j in range(A[1], B[1]) :
           for i in range(A[0], B[0]) :
                x = (i+0.5)/self.width * 2.0 - 1
                y = (j+0.5)/self.height * 2.0 - 1

                p = np.array([x,y])

                area0 = edgeSide(p,v0,v1)
                area1 = edgeSide(p,v1,v2)
                area2 = edgeSide(p,v2,v0)

                #test if p is inside the triangle
                if (area0 >= 0 and area1 >= 0 and area2 >= 0) :

                    #Computing 2d barricentric coordinates
                    lambda0 = area1/area
                    lambda1 = area2/area
                    lambda2 = area0/area

                    #one_over_z = lambda0 * 1/v0[2] + lambda1 * 1/v1[2] + lambda2 * 1/v2[2]
                    #z = 1/one_over_z

                    z = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]

                    p = np.array([x,y,z])


                    l = v0.shape[0]
                    #interpolating
                    interpolated_data = v0[3:l] * lambda0 + v1[3:l] * lambda1 + v2[3:l] * lambda2

                    #Emiting Fragment
                    fragments.append(Fragment(i, j, z, interpolated_data, lod,
                                              dudx, dvdx, dudy, dvdy))

        return fragments

    def fragmentShader(self,fragment,data):
        N = fragment.interpolated_data[0:3]
        N = N/np.linalg.norm(N)
        V = fragment.interpolated_data[3:6]
        V = V/np.linalg.norm(V)
        L = fragment.interpolated_data[6:9]
        L = L/np.linalg.norm(L)

        R = 2 * np.dot(L,N) * N  -L



        ambient = 1.0
        diffuse = max(np.dot(N,L),0)
        specular = np.power(max(np.dot(R,V),0.0),64)

        ka = 0.1
        kd = 0.9
        ks = 0.3
        phong = ka * ambient + kd * diffuse + ks * specular
        phong = np.ceil(phong*4 +1 )/6.0

        u = fragment.interpolated_data[9]
        v = fragment.interpolated_data[10]

        fm = self.filter_mode

        if fm == "nearest":
            lvl = min(int(np.floor(fragment.lod)), len(self.mips) - 1)
            tex_color = sample_nearest(self.mips[lvl], u, v)
        elif fm == "bilinear":
            lvl = min(int(np.floor(fragment.lod)), len(self.mips) - 1)
            tex_color = sample_bilinear(self.mips[lvl], u, v)
        elif fm == "trilinear":
            tex_color = sample_trilinear(self.mips, u, v, fragment.lod)
        elif fm == "anisotropic":
            tex_color = sample_anisotropic(
                self.mips, u, v, fragment.lod,
                fragment.dudx, fragment.dvdx,
                fragment.dudy, fragment.dvdy,
                max_samples=8)
        else:
            tex_color = sample_trilinear(self.mips, u, v, fragment.lod)

        color = np.array([phong,phong,phong]) * tex_color

        fragment.output = color

    def draw(self, vertices, triangles, data):
        self.mips = build_mipmaps(data['texture'], self.downsample_filter)

        #Calling vertex shader
        self.newVertices = np.zeros((vertices.shape[0], 18))

        for i in range(vertices.shape[0]) :
            self.newVertices[i] = self.VertexShader(vertices[i],data)

        fragments = []
        #Calling Rasterizer
        for i in triangles :
            fragments.extend(self.Rasterizer(self.newVertices[i[0]], self.newVertices[i[1]], self.newVertices[i[2]]))

        for f in fragments:
            self.fragmentShader(f,data)
            #depth test
            if self.depthBuffer[f.y][f.x] > f.depth :
                self.depthBuffer[f.y][f.x] = f.depth

                self.image[f.y][f.x] = f.output
