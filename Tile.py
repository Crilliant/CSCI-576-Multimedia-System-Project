class Tile:
    def __init__(self, corners=None, edges=None):
        if edges is None:
            edges = []
        if corners is None:
            corners = []
        self.corners = corners
        self.edges = edges
