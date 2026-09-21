package transport

import (
	"fmt"

	pb "github.com/RyanJHamby/distributed-gpu-training-flight-recorder/api/proto"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// ToProto converts a domain event to its wire form.
func ToProto(e types.Event) (*pb.EventMessage, error) {
	m := &pb.EventMessage{TimestampNs: e.TimestampNs(), Rank: e.Rank()}
	set := func(b types.BaseEvent) { m.NodeId, m.GpuUuid = b.NodeID, b.GPUUUID }
	switch v := e.(type) {
	case types.GPUMetricEvent:
		set(v.BaseEvent)
		m.Event = &pb.EventMessage_GpuMetric{GpuMetric: &pb.GPUMetric{
			TemperatureC: v.Temperature, PowerWatts: v.PowerWatts, UtilizationPct: v.Utilization,
			MemBandwidthPct: v.MemBandwidth, EccErrorsSbe: v.ECCErrorsSBE, EccErrorsDbe: v.ECCErrorsDBE,
			SmClockMhz: v.SMClockMHz, ThrottleReasons: v.ThrottleReasons}}
	case types.NCCLCollectiveEvent:
		set(v.BaseEvent)
		m.Event = &pb.EventMessage_NcclCollective{NcclCollective: &pb.NCCLCollective{
			PgId: v.PGID, SeqId: v.SeqID, Step: v.Step, OpType: v.OpType, DataSize: v.DataSize,
			DurationNs: v.DurationNs, Algorithm: v.Algorithm}}
	case types.PCIeBandwidthEvent:
		set(v.BaseEvent)
		m.Event = &pb.EventMessage_PcieBandwidth{PcieBandwidth: &pb.PCIeBandwidth{
			ReadBwMbs: v.ReadBandwidthMBs, WriteBwMbs: v.WriteBandwidthMBs}}
	case types.NVLinkEvent:
		set(v.BaseEvent)
		m.Event = &pb.EventMessage_Nvlink{Nvlink: &pb.NVLink{LinkId: v.LinkID, ThroughputGbs: v.ThroughputGB}}
	case types.ThermalEvent:
		set(v.BaseEvent)
		m.Event = &pb.EventMessage_Thermal{Thermal: &pb.Thermal{TemperatureC: v.TemperatureC, ThrottleActive: v.ThrottleActive}}
	default:
		return nil, fmt.Errorf("unsupported event type %T", e)
	}
	return m, nil
}

// FromProto converts a wire message back to a domain event. A message with no
// payload is an error rather than a silent zero event.
func FromProto(m *pb.EventMessage) (types.Event, error) {
	b := types.BaseEvent{Timestamp: m.TimestampNs, RankID: m.Rank, NodeID: m.NodeId, GPUUUID: m.GpuUuid}
	switch v := m.Event.(type) {
	case *pb.EventMessage_GpuMetric:
		g := v.GpuMetric
		return types.GPUMetricEvent{BaseEvent: b, Temperature: g.TemperatureC, PowerWatts: g.PowerWatts,
			Utilization: g.UtilizationPct, MemBandwidth: g.MemBandwidthPct, ECCErrorsSBE: g.EccErrorsSbe,
			ECCErrorsDBE: g.EccErrorsDbe, SMClockMHz: g.SmClockMhz, ThrottleReasons: g.ThrottleReasons}, nil
	case *pb.EventMessage_NcclCollective:
		n := v.NcclCollective
		return types.NCCLCollectiveEvent{BaseEvent: b, PGID: n.PgId, SeqID: n.SeqId, Step: n.Step,
			OpType: n.OpType, DataSize: n.DataSize, DurationNs: n.DurationNs, Algorithm: n.Algorithm}, nil
	case *pb.EventMessage_PcieBandwidth:
		return types.PCIeBandwidthEvent{BaseEvent: b, ReadBandwidthMBs: v.PcieBandwidth.ReadBwMbs,
			WriteBandwidthMBs: v.PcieBandwidth.WriteBwMbs}, nil
	case *pb.EventMessage_Nvlink:
		return types.NVLinkEvent{BaseEvent: b, LinkID: v.Nvlink.LinkId, ThroughputGB: v.Nvlink.ThroughputGbs}, nil
	case *pb.EventMessage_Thermal:
		return types.ThermalEvent{BaseEvent: b, TemperatureC: v.Thermal.TemperatureC, ThrottleActive: v.Thermal.ThrottleActive}, nil
	}
	return nil, fmt.Errorf("event message has no payload")
}
