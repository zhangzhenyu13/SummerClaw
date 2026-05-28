/** Dashboard layout — Ant Design Sider navigation with header. */

import React from 'react';
import { Layout as AntLayout, Menu, Typography } from 'antd';
import {
  UnorderedListOutlined,
  PlusCircleOutlined,
  DashboardOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';

const { Header, Sider, Content } = AntLayout;
const { Title } = Typography;

const MENU_ITEMS = [
  {
    key: '/',
    icon: <UnorderedListOutlined />,
    label: 'Task List',
  },
  {
    key: '/create',
    icon: <PlusCircleOutlined />,
    label: 'Create Task',
  },
];

export const DashboardLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Determine selected menu key from path
  const selectedKey = location.pathname === '/create' ? '/create' : '/';

  return (
    <AntLayout style={{ minHeight: '100vh' }}>
      <Sider
        breakpoint="lg"
        collapsedWidth={60}
        style={{ background: '#001529' }}
      >
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '0 12px',
          }}
        >
          <DashboardOutlined style={{ color: '#fff', fontSize: 24, marginRight: 8 }} />
          <Title level={5} style={{ color: '#fff', margin: 0, whiteSpace: 'nowrap' }}>
            Agent Trainer
          </Title>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={MENU_ITEMS}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <AntLayout>
        <Header
          style={{
            background: '#fff',
            padding: '0 24px',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <Title level={4} style={{ margin: 0 }}>
            Agent Trainer Dashboard
          </Title>
        </Header>
        <Content style={{ padding: 24, background: '#f5f5f5' }}>
          <Outlet />
        </Content>
      </AntLayout>
    </AntLayout>
  );
};
