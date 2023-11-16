from typing import List
import io
import os
import requests
import re
import argparse
import time

from typing import Sequence
from typing import Optional
from typing import Dict
from typing import Iterable
from typing import Any

from dataclasses import dataclass

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.cloud import storage

import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io.textio import WriteToText
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult

gcs_uri = "gs://sttfiles7/input"
gcs_output_path = "gs://sttfiles7/output"
gcs_bucket= "sttfiles7"
gcs_output_object = "output"
project_id = "bq-project-402513"

gcs_files = [
    "gs://sttfiles7/input/harvard.wav",
    "gs://sttfiles7/input/harvard1.wav"
]

@dataclass
class SpeechToTextClients:
    speech_client: SpeechClient
    storage_client: storage.Client


class RemoteModelHandler(ModelHandler[str,
                                     cloud_speech.BatchRecognizeResults,
                                     SpeechToTextClients]):
    def __init__(
        self,
    ):
        """
        Currently no-op, but probably should take in the endpoint info you're using to construct the request
        """
        pass

    def load_model(self) -> SpeechToTextClients:
        """Loads and initializes our remote inference clients."""
        return SpeechToTextClients(SpeechClient(), storage.Client())

    def run_inference(
        self,
        batch: Sequence[str],
        model: SpeechToTextClients,
        inference_args: Optional[Dict[str, Any]] = None
        ) -> List[Dict]: #str: #cloud_speech.BatchRecognizeResults:
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="long",
        )
        print(f"batch is ",batch)
        print(f"gcs_uri is ",gcs_uri)
        data = {}
        for file_uri in batch:
          file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=file_uri)

          request = cloud_speech.BatchRecognizeRequest(
              recognizer=f"projects/{project_id}/locations/global/recognizers/_",
              config=config,
              #files=batch,
              files=[file_metadata],
              recognition_output_config=cloud_speech.RecognitionOutputConfig(
                  gcs_output_config=cloud_speech.GcsOutputConfig(
                      uri=gcs_output_path,
                  ),
              ),
          )

          # Transcribes the audio into text
          operation = model.speech_client.batch_recognize(request=request)

          print("Waiting for operation to complete...")
          response = operation.result(timeout=120)
          print(f"response is", response)
          file_results = response.results[file_uri]

          print(f"Operation finished. Fetching results from {file_results.uri}...")
          output_bucket, output_object = re.match(
              r"gs://([^/]+)/(.*)", file_results.uri
          ).group(1, 2)

          #file_name = 'gs://sttfiles7/input/harvard.wav'
          file_name = file_results.uri.split('/')[-1].split('.')[0].split('_')[0]


          print(f"output_bucket is {output_bucket}")
          print(f"output_object is {output_object}")
          print(f"file uri is {file_results.uri}")
          print(f"file name is {file_name}")
          # Fetch results from Cloud Storage
          bucket = model.storage_client.bucket(output_bucket)
          blob = bucket.blob(output_object)
          results_bytes = blob.download_as_bytes()
          batch_recognize_results = cloud_speech.BatchRecognizeResults.from_json(
              results_bytes, ignore_unknown_fields=True
          )
          strlines = ""
          for result in batch_recognize_results.results:
            #print(f"Transcript: {result.alternatives[0].transcript}")
            strlines += result.alternatives[0].transcript
            print(f"strlines: {strlines}")
          #print(f"file_uri is",file_uri)
          data[file_name] = strlines
          print(f"data is {data}")
          return [data]

class GcsWrite(beam.DoFn):
  def setup(self):
    self._client = storage.Client()

  def process(self, file_name_contents_tuple):
    tuple = list(file_name_contents_tuple)
    k1 = tuple[0]
    v1 = file_name_contents_tuple[k1]
    #print(f"key is ",k1)
    #print(f"value is ",file_name_contents_tuple[k1])

    # file_name_contents_tuple is whatever you outputed from the run_inference function
    bucket = self._client.bucket(gcs_bucket)
    file_path = gcs_output_object + "/" + k1 + ".txt"
    blob = bucket.blob(file_path)
    with blob.open("w") as f:
        f.write(v1)

parser = argparse.ArgumentParser()
parser.add_argument('--project',required=True, help='Specify Google Cloud project')
parser.add_argument('--region', required=True, help='Specify Google Cloud region')
parser.add_argument('--stagingLocation', required=True, help='Specify Cloud Storage bucket for staging')
parser.add_argument('--tempLocation', required=True, help='Specify Cloud Storage bucket for temp')
parser.add_argument('--runner', required=True, help='Specify Apache Beam Runner')
parser.add_argument('--requirements_file', required=True, help='Specify requirements file')
#parser.add_argument('--job_name', required=True, help='Specify job name')

opts = parser.parse_args()

# Setting up the Beam pipeline options
options = PipelineOptions()
options.view_as(GoogleCloudOptions).project = opts.project
options.view_as(GoogleCloudOptions).region = opts.region
options.view_as(GoogleCloudOptions).staging_location = opts.stagingLocation
options.view_as(GoogleCloudOptions).temp_location = opts.tempLocation
options.view_as(GoogleCloudOptions).job_name = '{0}{1}'.format('my-pipeline-',time.time_ns())
options.view_as(StandardOptions).runner = opts.runner
options.view_as(SetupOptions).requirements_file = opts.requirements_file
#options.view_as(GoogleCloudOptions).job_name = opts.job_name


pipeline = beam.Pipeline(options=options)
(pipeline | "Create inputs" >> beam.Create(gcs_files)
 | "Inference" >> RunInference(RemoteModelHandler())
 | "Write Files" >> beam.ParDo(GcsWrite()))

pipeline.run()
