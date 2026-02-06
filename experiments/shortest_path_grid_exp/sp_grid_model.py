from decision_learning.benchmarks.shortest_path_grid.oracle import opt_oracle

# optimization model (commented out; legacy PyEPO/Pyomo integration)
# class optGenModel(optModel):
#     """
#     This is an abstract class for Pyomo-based optimization model
#
#     Attributes:
#         _model (PyOmo model): Pyomo model
#         solver (str): optimization solver in the background
#     """
#
#     def __init__(self):
#         """
#         Args:
#             solver (str): optimization solver in the background
#         """
#         super().__init__()
#         # init obj
#         if self._model.modelSense == EPO.MINIMIZE:
#             self.modelSense = EPO.MINIMIZE
#         if self._model.modelSense == EPO.MAXIMIZE:
#             self.modelSense = EPO.MAXIMIZE
#
#     def __repr__(self):
#         return "optGenModel " + self.__class__.__name__
#
#     def setObj(self, c):
#         """
#         A method to set objective function
#
#         Args:
#             c (np.ndarray / list): cost of objective function
#         """
#         self._model.costvec = c
#
#     def copy(self):
#         """
#         A method to copy model
#
#         Returns:
#             optModel: new copied model
#         """
#         new_model = copy(self)
#         return new_model
#
#     def addConstr(self):
#         new_model = self.copy()
#         # add constraint
#         return new_model
#
# class modelclass():
#     def __init__(self, size):
#         self.size = size
#         self.costvec = None
#         self.modelSense = EPO.MINIMIZE
#         self.x = np.ones(2 * size * (size - 1))
# class shortestPathModel(optGenModel):
#
#     def __init__(self):
#         self.grid = (5,5)
#         super().__init__()
#
#     def _getModel(self):
#         """
#         A method to build Gurobi model
#
#         Returns:
#             tuple: optimization model and variables
#         """
#         m = modelclass(self.grid[0])
#         x = m.x
#         # sense
#         m.modelSense = EPO.MINIMIZE
#         return m, x
#
#     def solve(self):
#         sol, obj = opt_oracle(self._model.costvec.reshape(-1,len(self.x)), self._model.size)
#         return sol, obj
