class Tile:
    def __init__(self, corners=None, edges=None, image=None, rotation=0):
        if edges is None:
            edges = []
        if corners is None:
            corners = []
        self.corners = corners
        self.edges = edges
        self.image = image  
        self.rotation = rotation
        self.position = None 

        # Parameters for the animation
        self.initial_position = (0, 0)
        self.final_position = (0, 0)
        self.initial_rotation = 0
        self.final_rotation = 0
        self.extracted_img = None  # 裁剪出的拼图块图像数据 (np.ndarray)
