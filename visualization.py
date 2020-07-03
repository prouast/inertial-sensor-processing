import pygame
import quaternion
import numpy as np
from operator import itemgetter

class Node:
  """A node is an edge of the cuboid"""
  def __init__(self, coords, color):
    self.x = coords[0]
    self.y = coords[1]
    self.z = coords[2]
    self.color = color

class Face:
  """A face of the cuboid is defined using the indices of four nodes"""
  def __init__(self, nodeIdxs, color):
    self.nodeIdxs = nodeIdxs
    self.color = color

class Cuboid:
  """The cuboid"""
  def __init__(self, quaternion):
    self.nodes = []
    self.faces = []
    self.q = quaternion

  def set_nodes(self, nodes):
    self.nodes = nodes

  def set_faces(self, faces):
    self.faces = faces

  def set_quaternion(self, q):
    self.q = q

  def rotate_quaternion(self, w, dt):
    self.q = dt/2 * self.q * np.quaternion(0, w[0], w[1], w[2]) + self.q

  def rotate_point(self, point):
    return quaternion.rotate_vectors(self.q, point)

  def convert_to_computer_frame(self, point):
    computerFrameChangeMatrix = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    return np.matmul(computerFrameChangeMatrix, point)

  def get_euler_attitude(self):
    def _rad2deg(rad):
      return rad / np.pi * 180
    m = quaternion.as_rotation_matrix(self.q)
    test = -m[2, 0]
    if test > 0.99999:
      yaw = 0
      pitch = np.pi / 2
      roll = np.arctan2(m[0, 1], m[0, 2])
    elif test < -0.99999:
      yaw = 0
      pitch = -np.pi / 2
      roll = np.arctan2(-m[0, 1], -m[0, 2])
    else:
      yaw = np.arctan2(m[1, 0], m[0, 0])
      pitch = np.arcsin(-m[2, 0])
      roll = np.arctan2(m[2, 1], m[2, 2])
    yaw = _rad2deg(yaw)
    pitch = _rad2deg(pitch)
    roll = _rad2deg(roll)
    return yaw, pitch, roll

class PygameViewer:
  """Displays 3D objects on a Pygame screen"""

  def __init__(self, width, height, quaternion, loopRate):
    self.width = width
    self.height = height
    self.cuboid = self.initialize_cuboid(quaternion)
    self.loopRate = loopRate
    self.screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Sensor fusion for inertial sensors')
    self.background = (10,10,50)
    self.clock = pygame.time.Clock()
    pygame.font.init()
    self.font = pygame.font.SysFont('Comic Sans MS', 20)

  def initialize_cuboid(self, quaternion):
    """Initialize cuboid with nodes and faces"""
    # The cuboid with initial quaternion
    cuboid = Cuboid(quaternion)
    # Define nodes
    nodes = []
    nodes.append(Node([-1.5, -1, -0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, -1, 0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, 1, -0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, 1, 0.1], [255, 255, 255]))
    nodes.append(Node([1.5, -1, -0.1], [255, 255, 255]))
    nodes.append(Node([1.5, -1, 0.1], [255, 255, 255]))
    nodes.append(Node([1.5, 1, -0.1], [255, 255, 255]))
    nodes.append(Node([1.5, 1, 0.1], [255, 255, 255]))
    cuboid.set_nodes(nodes)
    # Define faces
    faces = []
    faces.append(Face([0, 2, 6, 4], [255, 0, 255]))
    faces.append(Face([0, 1, 3, 2], [255, 0, 0]))
    faces.append(Face([1, 3, 7, 5], [0, 255, 0]))
    faces.append(Face([4, 5, 7, 6], [0, 0, 255]))
    faces.append(Face([2, 3, 7, 6], [0, 255, 255]))
    faces.append(Face([0, 1, 5, 4], [255, 255, 0]))
    cuboid.set_faces(faces)
    return cuboid

  def set_quaternion(self, q):
    self.cuboid.set_quaternion(q)

  def update(self):
    """Update the screen"""
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return False
    self.clock.tick(self.loopRate)
    self.display()
    pygame.display.flip()
    return True

  def display(self):
    """Draw the cuboid on the screen."""
    self.screen.fill(self.background)

    # Display the current attitude
    yaw, pitch, roll = self.cuboid.get_euler_attitude()
    self.message_display("Yaw: %.1f" % yaw,
              self.screen.get_width()*0.75,
              self.screen.get_height()*0,
              (220, 20, 60))    # Crimson
    self.message_display("Pitch: %.1f" % pitch,
              self.screen.get_width()*0.75,
              self.screen.get_height()*0.05,
              (0, 255, 255))   # Cyan
    self.message_display("Roll: %.1f" % roll,
              self.screen.get_width()*0.75,
              self.screen.get_height()*0.1,
              (65, 105, 225))  # Royal Blue

    # Transform nodes to perspective view
    dist = 5
    pvNodes = []
    pvDepth = []
    for node in self.cuboid.nodes:
      point = [node.x, node.y, node.z]
      newCoord = self.cuboid.rotate_point(point)
      comFrameCoord = self.cuboid.convert_to_computer_frame(newCoord)
      pvNodes.append(self.project_othorgraphic(comFrameCoord[0], comFrameCoord[1], comFrameCoord[2],
        self.screen.get_width(), self.screen.get_height(), 70, pvDepth))
      #pvDepth.append(node.z)

    # Calculate the average Z values of each face.
    avg_z = []
    for face in self.cuboid.faces:
      n = pvDepth
      z = (n[face.nodeIdxs[0]] + n[face.nodeIdxs[1]] +
         n[face.nodeIdxs[2]] + n[face.nodeIdxs[3]]) / 4.0
      avg_z.append(z)
    # Draw the faces using the Painter's algorithm:
    for idx, val in sorted(enumerate(avg_z), key=itemgetter(1)):
      face = self.cuboid.faces[idx]
      pointList = [pvNodes[face.nodeIdxs[0]],
             pvNodes[face.nodeIdxs[1]],
             pvNodes[face.nodeIdxs[2]],
             pvNodes[face.nodeIdxs[3]]]
      pygame.draw.polygon(self.screen, face.color, pointList)

  def project_one_point_perspective(self, x, y, z, win_width, win_height, P, S, scaling_constant, pvDepth):
    """One vanishing point perspective view algorithm"""
    # In Pygame, the y axis is downward pointing.
    # In order to make y point upwards, a rotation around x axis by 180 degrees is needed.
    # This will result in y' = -y and z' = -z
    xPrime = x
    yPrime = -y
    zPrime = -z
    xProjected = xPrime * (S / (zPrime + P)) * scaling_constant + win_width / 2
    yProjected = yPrime * (S / (zPrime + P)) * scaling_constant + win_height / 2
    pvDepth.append(1 / (zPrime + P))
    return (round(xProjected), round(yProjected))

  def project_othorgraphic(self, x, y, z, win_width, win_height, scaling_constant, pvDepth):
    """Normal Projection"""
    # In Pygame, the y axis is downward pointing.
    # In order to make y point upwards, a rotation around x axis by 180 degrees is needed.
    # This will result in y' = -y and z' = -z
    xPrime = x
    yPrime = -y
    xProjected = xPrime * scaling_constant + win_width / 2
    yProjected = yPrime * scaling_constant + win_height / 2
    # Note that there is no negative sign here because our rotation to computer frame
    # assumes that the computer frame is x-right, y-up, z-out
    # so this z-coordinate below is already in the outward direction
    pvDepth.append(z)
    return (round(xProjected), round(yProjected))

  def message_display(self, text, x, y, color):
    textSurface = self.font.render(text, True, color, self.background)
    textRect = textSurface.get_rect()
    textRect.topleft = (x, y)
    self.screen.blit(textSurface, textRect)
