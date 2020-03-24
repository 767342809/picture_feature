import tensorflow as tf
from tensorflow.python.platform import gfile
import os
from tensorflow.python.tools import freeze_graph
import numpy as np
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import vgg, resnet_v1, inception, resnet_v2
# model_path = "/Users/liangyue/Documents/frozen_model_vgg_16/model/vgg_16.ckpt"  # 设置model的路径
# model_name = "vgg16"
# builder_result_path = "./vgg16"

model_path = "/Users/liangyue/Documents/frozen_model_vgg_16/model/resnet_v1_50.ckpt"
model_name = "resnet_v1_50model"
builder_result_path = "./resnet_v1_50"

# model_path = "/Users/liangyue/Documents/frozen_model_vgg_16/model/inception_v3.ckpt"
# model_name = "inception_v3model"
# builder_result_path = "./inception_v3"

# model_path = "/Users/liangyue/Documents/frozen_model_vgg_16/model/resnet_50_ILSVRC_iNat_299.ckpt"
# model_name = "resnet_v2_50model"
# builder_result_path = "./resnet_v2_50"


def main():
    tf.reset_default_graph()

    # 240, 240, 3
    input_node = tf.placeholder(tf.float32, shape=(299, 299, 3))
    input_node = tf.expand_dims(input_node, 0)

    # # vgg16
    # with slim.arg_scope(vgg.vgg_arg_scope()):
    #     flow, _ = vgg.vgg_16(input_node)
    # flow = tf.cast(flow, tf.uint8, 'out')  # 设置输出类型以及输出的接口名字，为了之后的调用pb的时候使用

    # # resnet_v1_50
    # with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    #     flow, _ = resnet_v1.resnet_v1_50(input_node, 1000)
    # flow = tf.cast(flow, tf.uint8, 'out')

    # ## inception
    # with slim.arg_scope(inception.inception_v3_arg_scope()):
    #     flow, _ = inception.inception_v3(input_node)
    # flow = tf.cast(flow, tf.uint8, 'out')

    # resnet_v2_50
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        flow, _ = resnet_v2.resnet_v2_50(input_node, is_training=False)
    flow = tf.cast(flow, tf.uint8, 'out')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        # 保存图
        tf.train.write_graph(sess.graph_def, f'./output_model/{model_name}', "model.pb")
        # 把图和参数结构一起
        freeze_graph.freeze_graph(f'output_model/{model_name}/model.pb',
                                  '',
                                  False,
                                  model_path,
                                  'out',
                                  'save/restore_all',
                                  'save/Const:0',
                                  f'output_model/{model_name}/{model_name}.pb',
                                  False,
                                  "")

        saver.restore(sess, model_path)
        builder = tf.saved_model.builder.SavedModelBuilder(builder_result_path)
        builder.add_meta_graph_and_variables(sess, ["serve"])
        builder.save()

    print("done")


def load_pb():
    sess = tf.Session()
    model_1 = f'output_model/{model_name}/{model_name}.pb'
    model_2 = f'{builder_result_path}/saved_model.pb'

    # first
    with gfile.FastGFile(model_1, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    constant_ops = [op for op in sess.graph.get_operations()]
    for constant_op in constant_ops:
        print(constant_op.name)


if __name__ == '__main__':
    # main()
    load_pb()