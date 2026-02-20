class Section:
    """
    表示截面属性。
    """
    def __init__(self, A: float, I: float = 0.0):
        """
        :param A: 截面积 (Cross-sectional Area)
        :param I: 惯性矩 (Moment of Inertia)
        """
        self.A: float = A
        self.I: float = I  # 桁架单元可以不用 I，但梁单元必须有

    def __repr__(self) -> str:
        return f"Section(A={self.A}, I={self.I})"
