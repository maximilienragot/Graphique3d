import numpy as np

def readply(filename) :
  file = open(filename, 'r')
  lines = file.readlines()
  vertices = []
  triangles = []

  nbVertices = 0
  nbFace = 0
  state = 0
  counter = 0
  for line in lines :

    if state == 0 :
      line = line.rstrip().split(' ')
      if line[0] == 'element' :
        if line[1] == 'vertex' :
          nbVertices = int(line[2])
        if line[1] == 'face' :
          nbFace = int(line[2])

      if line[0] == 'end_header' :
        state = 1
        continue

    if state == 1 :
      line = line.split(' ')
      vertex = []
      for l in line:
        vertex.append(float(l))
      vertices.append(vertex)
      counter = counter + 1
      if counter == nbVertices:
        counter = 0
        state = 2
        continue

    if state == 2 :
      line = line.split(' ')
      line.pop(0)
      triangle =[]
      for l in line:
        triangle.append(int(l))
      triangles.append(triangle)

  return np.array(vertices),np.array(triangles,dtype=int)
