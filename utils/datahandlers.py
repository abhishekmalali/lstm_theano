import os

class DataHandler():

    def __init__(self):
        return

    def _create_graph_path(self, base_graph_path, folder_name):
        # Helper Function
        graphs_path = os.path.join(os.getcwd(), base_graph_path)
        store_path = os.path.join(graphs_path, folder_name)
        pred_path = os.path.join(store_path, 'predicted-graphs')
        if not os.path.isdir(pred_path):
            os.makedirs(pred_path)
        pred_path = pred_path + '/'
        return pred_path

    def _create_error_path(self, folder_name):
        # Helper Function
        graphs_path = os.path.join(os.getcwd(), self.base_graph_path)
        store_path = os.path.join(graphs_path, folder_name)
        error_path = os.path.join(store_path, 'error-graphs')
        if not os.path.isdir(error_path):
            os.makedirs(error_path)
        error_path = error_path + '/'
        return error_path

    def create_paths(self, base_graph_path, folder_name):
        pred_path = self._create_graph_path(base_graph_path, folder_name)
        self.base_graph_path = base_graph_path
        error_path = self._create_error_path(folder_name)
        self.pred_path = pred_path
        self.error_path = error_path
        return pred_path, error_path

    def create_new_pred_path(self, folder_name):
        # To create new paths within predicted-graphs
        new_path = os.path.join(self.pred_path, folder_name)
        if not os.path.isdir(new_path):
            os.makedirs(new_path)
        new_path = new_path + '/'
        return new_path
