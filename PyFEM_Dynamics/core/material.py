class Material:
    """
    表示材料属性。
    """
    def __init__(self, E: float, rho: float):
        """
        :param E: 弹性模量 (Elastic Modulus)
        :param rho: 密度 (Density)
        """
        self.E: float = E
        self.rho: float = rho

    def __repr__(self) -> str:
        return f"Material(E={self.E}, rho={self.rho})"
