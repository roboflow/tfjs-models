/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {CLASSES} from './classes';

const BASE_PATH = 'https://storage.googleapis.com/tfjs-models/savedmodel/';

export {version} from './version';

export type ObjectDetectionBaseModel =
    'mobilenet_v1'|'mobilenet_v2'|'lite_mobilenet_v2';

export interface DetectedObject {
  bbox: [number, number, number, number];  // [x, y, width, height]
  class: string;
  score: number;
}

/**
 * Coco-ssd model loading is configurable using the following config dictionary.
 *
 * `base`: ObjectDetectionBaseModel. It determines wich PoseNet architecture
 * to load. The supported architectures are: 'mobilenet_v1', 'mobilenet_v2' and
 * 'lite_mobilenet_v2'. It is default to 'lite_mobilenet_v2'.
 *
 * `modelUrl`: An optional string that specifies custom url of the model. This
 * is useful for area/countries that don't have access to the model hosted on
 * GCP.
 */
export interface ModelConfig {
  base?: ObjectDetectionBaseModel;
  modelUrl?: string;
}

export async function load(config: ModelConfig = {}) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  const base = config.base || 'lite_mobilenet_v2';
  const modelUrl = config.modelUrl;
  // if (['mobilenet_v1', 'mobilenet_v2', 'lite_mobilenet_v2'].indexOf(base) ===
  //     -1) {
  //   throw new Error(
  //       `ObjectDetection constructed with invalid base model ` +
  //       `${base}. Valid names are 'mobilenet_v1',` +
  //       ` 'mobilenet_v2' and 'lite_mobilenet_v2'.`);
  // }

  const objectDetection = new ObjectDetection(base, modelUrl);
  await objectDetection.load(base);
  return objectDetection;
}

export class ObjectDetection {
  private modelPath: string;
  private model: tfconv.GraphModel;


  //private zeros:tf.Tensor3D = tf.zeros([640, 640, 3], 'float32');
  //private zeros:tf.Tensor3D = tf.randomUniform([416, 416, 3], 0, 255, 'float32');

  constructor(base: ObjectDetectionBaseModel, modelUrl?: string) {
    this.modelPath =
        modelUrl || `${BASE_PATH}${this.getPrefix(base)}/model.json`;
  }

  private getPrefix(base: ObjectDetectionBaseModel) {
    return base === 'lite_mobilenet_v2' ? `ssd${base}` : `ssd_${base}`;
  }

  async load(base:any) {
    this.model = await tfconv.loadGraphModel(this.modelPath);

    console.log('Starting Zero Tensor');

    const zeroTensor = (base === 'yolov5s') ? tf.zeros([1, 416, 416, 3], 'float32') : tf.zeros([1, 300, 300, 3], 'int32');

    // Warmup the model.
    const result = await this.model.executeAsync(zeroTensor) as tf.Tensor[];
    await Promise.all(result.map(t => t.data()));
    result.map(t => t.dispose());
    zeroTensor.dispose();
  }

  /**
   * Infers through the model.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   * @param minScore The minimum score of the returned bounding boxes
   * of detected objects. Value between 0 and 1. Defaults to 0.5.
   * @param nmsThresh Threshold for filtering overlapping boxes

   */
  private async infer(
      base:any,
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      maxNumBoxes: number,
      nmsThresh: number,
      minScore: number): Promise<DetectedObject[]> {


    const batched = tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        console.log(img.height); //this confirms the video is 640x640

        img = (base === 'yolov5s') ? tf.browser.fromPixels(img).resizeNearestNeighbor([416, 416]).asType('float32') : tf.browser.fromPixels(img); //img is now 480x640
        //img = (base === 'yolov5s') ? tf.randomUniform([640, 640, 3], 0, 255, 'float32') : tf.browser.fromPixels(img); //img is now 480x640
        console.log('img.shape', img.shape);
      }
      // Reshape to a single-element batch so we can pass it to executeAsync.
      //img = img.reshape([640,640]);
      return img.expandDims(0);
    });
    const height = batched.shape[1];
    const width = batched.shape[2];


    // model returns two tensors:
    // 1. box classification score with shape of [1, 1917, 90]
    // 2. box location with shape of [1, 1917, 1, 4]
    // where 1917 is the number of box detectors, 90 is the number of classes.
    // and 4 is the four coordinates of the box.
    console.log("BEFORE INFER");
    var t0 = performance.now();

    const result = await this.model.executeAsync(batched) as tf.Tensor[];
    //const result = this.model.execute(batched) as tf.Tensor[];

    //const result = await this.model.executeAsync(batched) as tf.Tensor[];
    var t1 = performance.now();
    console.log("Call to INFER took " + (t1 - t0) + " milliseconds.");

    // console.log('inference shape: ', result.shape)
    // console.log('mobilenet output tensor shape scores', result[0].shape);
    // console.log('mobilenet output tensor shape scores boxes,' result[1].shape);
    //
    // output tensor shape scores (3) [1, 1917, 90]
    // output tensor shape scores boxes, (4) [1, 1917, 1, 4]

    //yolov5 tensor
    //console.log('yolov5 output tensor shape', result[3].shape);
    //[1,10647,5+num_classes] 5+num_classes is x, y, width, height, confidence, class_conf...


    // console.log(result[1].shape);
    // console.log(result[2].shape);
    // console.log(result[3].shape);
    // console.log(result)



    console.log("AFTER INFER HAS OCURRED");

    const scores = result[0].dataSync() as Float32Array;
    const boxes = result[1].dataSync() as Float32Array;



    //const yolov5_scores = result[3];
    //const yolov5_scores_await = await yolov5_scores.buffer();
    //const yolov5_scores = yolov5_scores.get();
    // console.log('yolov5 scores buffer', yolov5_scores_await);
    //
    // console.log(result[3].shape[1]);
    // console.log(result[3].shape[2]);

    //console.log(result[3].dataSync() as Float32Array);


    //slice of tensor with each of the class probabilities


    //console.log('yolov5 scores slice shape', yolov5_scores_slice);
    if (base === 'yolov5s') {
        const yolov5_scores_slice = result[3].slice([0,0,5], [1,result[3].shape[1],result[3].shape[2]-5]).dataSync() as Float32Array;

        const [yolov5maxScores, yolov5classes] =
                this.calculateMaxScores(yolov5_scores_slice, result[3].shape[1],result[3].shape[2]-5);

        const yolov5_confidence = result[3].slice([0,0,4], [1,result[3].shape[1],1).dataSync() as Float32Array;
        //yolo confidence is seperately output - mobilenet must bake into class confidence prediction

        const x1 = tf.sub(result[3].slice([0,0,0], [1,result[3].shape[1],1), tf.div(result[3].slice([0,0,2], [1,result[3].shape[1],1), 2));
        const y1 = tf.sub(result[3].slice([0,0,1], [1,result[3].shape[1],1), tf.div(result[3].slice([0,0,3], [1,result[3].shape[1],1), 2));
        const x2 = tf.add(result[3].slice([0,0,0], [1,result[3].shape[1],1), tf.div(result[3].slice([0,0,2], [1,result[3].shape[1],1), 2));
        const y2 = tf.add(result[3].slice([0,0,1], [1,result[3].shape[1],1), tf.div(result[3].slice([0,0,3], [1,result[3].shape[1],1), 2));

        const yolov5_box_corners = tf.div(tf.concat([x1, y1, x2, y2], 2), 416).dataSync() as Float32Array;

        // clean the webgl tensors
        batched.dispose();
        tf.dispose(result);

        const prevBackend = tf.getBackend();
        // run post process in cpu
        tf.setBackend('cpu');

        const yolov5_indexTensor = tf.tidy(() => {

          // output tensor shape scores boxes, (4) [1, 1917, 1, 4] (x1,y1,x2,y2) normalized

          //console.log('x1', x1);
          console.log('yolov5_box_corners, ' yolov5_box_corners);


          const yolov5_boxes2 = tf.tensor2d(yolov5_box_corners, [result[3].shape[1], 4]);

          //boxes2 entries look like [0.0015282053500413895, 0.0011117402464151382, 0.04202847182750702, 0.061883047223091125, -0.07541173696517944, -0.07601074129343033
          //console.log('boxes2 shape', boxes2); (num objects, 4 box locations)
          //2d tesnsor of shape [num_objects, 4], 4 being the box locations (x1,y1,x2,y2) and normalized 0,1


          return tf.image.nonMaxSuppression(
              yolov5_boxes2, yolov5_confidence, maxNumBoxes, nmsThresh, minScore);
        });

        console.log('yolov5_indexTensor', yolov5_indexTensor);

        const yolov5_indexes = yolov5_indexTensor.dataSync() as Float32Array;
        yolov5_indexTensor.dispose();

        //console.log(indexes);
        //indexes are an array of indices of objects to be included after nms

        //if(Math.random() < 10) return null;

        // restore previous backend
        tf.setBackend(prevBackend);

        return this.buildDetectedObjects(
            width, height, yolov5_box_corners, yolov5_confidence, yolov5_indexes, yolov5classes);

    }

    else {
        // clean the webgl tensors
        batched.dispose();
        tf.dispose(result);

        const [maxScores, classes] =
            this.calculateMaxScores(scores, result[0].shape[1], result[0].shape[2]);

        // console.log('maxScores', maxScores);
        // console.log('classes tenosr', classes);
        //

        // seems to be working - though with cells everything is low conf and prob of red cell
        // console.log('yolov5maxScores', yolov5maxScores);
        // console.log('yolov5classes', yolov5classes);

        // const boxes2 =
        //     tf.tensor2d(boxes, [result[1].shape[1], result[1].shape[3
        // console.log('boxes2 shape', boxes2.shape);

        //if(Math.random() < 10) return null;

        const prevBackend = tf.getBackend();
        // run post process in cpu
        tf.setBackend('cpu');

        const indexTensor = tf.tidy(() => {
          const boxes2 = tf.tensor2d(boxes, [result[1].shape[1], result[1].shape[3]]);
          //console.log('boxes2 ', boxes2.dataSync() as Float32Array);

          //boxes2 entries look like [0.0015282053500413895, 0.0011117402464151382, 0.04202847182750702, 0.061883047223091125, -0.07541173696517944, -0.07601074129343033
          //console.log('boxes2 shape', boxes2); (num objects, 4 box locations)
          //2d tesnsor of shape [num_objects, 4], 4 being the box locations (x1,y1,x2,y2) and normalized 0,1


          return tf.image.nonMaxSuppression(
              boxes2, maxScores, maxNumBoxes, nmsThresh, minScore);
        });

        //console.log('indexTensor', indexTensor.dataSync() as Float32Array);
        //index tensor is a tensor with the indices of the objects to keep in the list (in the case of COCO 1)


        const indexes = indexTensor.dataSync() as Float32Array;
        indexTensor.dispose();

        //console.log(indexes);
        //indexes are an array of indices of objects to be included after nms

        //if(Math.random() < 10) return null;

        // restore previous backend
        tf.setBackend(prevBackend);

        return this.buildDetectedObjects(
            width, height, boxes, maxScores, indexes, classes);

    }


  }

  private buildDetectedObjects(
      width: number, height: number, boxes: Float32Array, scores: number[],
      indexes: Float32Array, classes: number[]): DetectedObject[] {
    const count = indexes.length;
    const objects: DetectedObject[] = [];
    for (let i = 0; i < count; i++) {
      const bbox = [];
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j];
      }
      const minY = bbox[0] * height;
      const minX = bbox[1] * width;
      const maxY = bbox[2] * height;
      const maxX = bbox[3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;
      objects.push({
        bbox: bbox as [number, number, number, number],
        class: CLASSES[classes[indexes[i]] + 1].displayName,
        score: scores[indexes[i]]
      });
    }
    return objects;
  }

  private calculateMaxScores(
      scores: Float32Array, numBoxes: number,
      numClasses: number): [number[], number[]] {
    const maxes = [];
    const classes = [];
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE;
      let index = -1;
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j];
          index = j;
        }
      }
      maxes[i] = max;
      classes[i] = index;
    }
    return [maxes, classes];
  }

  /**
   * Detect objects for an image returning a list of bounding boxes with
   * assocated class and score.
   *
   * @param img The image to detect objects from. Can be a tensor or a DOM
   *     element image, video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   * @param minScore The minimum score of the returned bounding boxes
   * of detected objects. Value between 0 and 1. Defaults to 0.5.
   */
  async detect(
      base:any,
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      maxNumBoxes = 20,
      nmsThresh = 0.5,
      minScore = 0.5): Promise<DetectedObject[]> {
    return this.infer(base, img, maxNumBoxes, nmsThresh, minScore);
  }

  /**
   * Dispose the tensors allocated by the model. You should call this when you
   * are done with the model.
   */
  dispose() {
    if (this.model != null) {
      this.model.dispose();
    }
  }
}
